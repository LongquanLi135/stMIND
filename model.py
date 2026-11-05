"""Full stMIND particle dynamics for regeneration and embryogenesis, seed 0."""
import math
import torch
import torchsde
from torch import nn
from torch.utils.checkpoint import checkpoint
from latentsde_layers import (FullAttention, PositionMLP, CellMassTurnoverDrift,
                              preserve_seed0_initialization)
from losses import centered_growth_variance, local_drift_penalty


class STMind(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_x, self.d_s = args.gene_dim, args.pos_dim
        self.d = self.d_x + self.d_s
        self.log_mass_index, self.energy_index = self.d, self.d + 1
        self.relative_growth_index, self.local_drift_index = self.d + 2, self.d + 3
        self.mass_energy_weight = args.mass_dynamics_energy_weight
        layers, width = [], self.d_x
        for hidden in [128, 256, 256, 128]:
            layers.extend([nn.Linear(width, hidden), nn.Softplus()])
            width = hidden
        layers.append(nn.Linear(width, self.d_x))
        self.intrinsic = nn.Sequential(*layers)
        nn.init.zeros_(self.intrinsic[-1].weight)
        nn.init.zeros_(self.intrinsic[-1].bias)
        # Reserve the archived state-dict ordering; these slots are replaced.
        self.interaction = nn.Identity()
        self.position = nn.Identity()
        preserve_seed0_initialization(self.d_x, self.d_s)
        self.growth = CellMassTurnoverDrift(self.d_x, self.d_s,
            args.mass_dynamics_rate_scale, args.mass_dynamics_time_scale)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.position_initialization_seed)
            self.position = PositionMLP(self.d_x, self.d_s, args.position_hidden_dim,
                args.position_hidden_layers, args.position_time_scale)
        self.dt = args.train_dt
        self.checkpoint_steps = args.checkpoint_steps
        self.sigma_x, self.sigma_s = args.sigma_const_gene, args.sigma_const_pos
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.gt_initialization_seed)
            self.interaction = FullAttention(args)

    def initial_state(self, row):
        return torch.cat([row['x'], row['s'], row['x'].new_zeros(len(row['x']), 4)], 1)


    def mass_from_log_mass(self, ell):
        return ell.exp()

    def drift_components(self, state, pool):
        x = state[:, :self.d_x]
        s = state[:, self.d_x:self.d]
        ell = state[:, self.log_mass_index]
        interaction = self.interaction(x, s, ell, pool, return_components=True)
        intrinsic = self.intrinsic(x)
        return dict(intrinsic=intrinsic, **interaction, expression_total=intrinsic + interaction['total'])

    def f(self, t, state, pool, graph=None):
        (x, s) = (state[:, :self.d_x], state[:, self.d_x:self.d])
        ell = state[:, self.log_mass_index:self.log_mass_index + 1]
        mass = self.mass_from_log_mass(ell)
        intrinsic = self.intrinsic(x)
        interaction = self.interaction(x, s, ell[:, 0], pool, graph=graph, return_components=True)
        bx = intrinsic + interaction['total']
        bs = self.position(t, x, s)
        rate = self.growth(t, x, s, torch.zeros_like(ell))
        action = bx.square().sum(1, keepdim=True) + bs.square().sum(1, keepdim=True)
        action = action + self.mass_energy_weight * rate.square()
        rg = centered_growth_variance(rate, ell)
        lg = local_drift_penalty(interaction['local_contribution'], ell)
        return torch.cat([bx, bs, rate, mass * action, rg.expand(len(state), 1), lg.expand(len(state), 1)], 1)

    def step(self, state, increment, time, dt, pool, graph):
        noise = torch.cat([self.sigma_x * increment[:, :self.d_x], self.sigma_s * increment[:, self.d_x:], state.new_zeros(len(state), 4)], 1)
        return state + dt * self.f(time, state, pool, graph) + noise

    def rollout(self, times, state0, pool, brownian_seed, dt=None, checkpoint_steps=None):
        times = list(map(float, times))
        dt = self.dt if dt is None else float(dt)
        if dt <= 0 or len(times) < 2 or any((b <= a for (a, b) in zip(times, times[1:]))):
            raise ValueError('Positive step and increasing times required')
        cp = self.checkpoint_steps if checkpoint_steps is None else checkpoint_steps
        cp = cp and torch.is_grad_enabled()
        bm = torchsde.BrownianInterval(t0=times[0], t1=times[-1], size=(len(state0), self.d), device=state0.device, dtype=state0.dtype, entropy=int(brownian_seed), halfway_tree=True, tol=1e-06)
        state = state0
        output = [state0]
        current = times[0]
        for target in times[1:]:
            n_steps = int(math.ceil((target - current) / dt - 1e-09))
            for _ in range(n_steps):
                nxt = min(current + dt, target)
                graph = self.interaction.build_graph(state[:, self.d_x:self.d])
                operands = (state, bm(current, nxt), state.new_tensor(current), state.new_tensor(nxt - current), pool, graph)
                state = checkpoint(self.step, *operands, use_reentrant=False) if cp else self.step(*operands)
                current = nxt
            output.append(state)
        return torch.stack(output)
