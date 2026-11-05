"""Active neural layers, exact radius graph and full mass-aware attention."""

import math
import torch
from torch import nn

class AttentionBranch(nn.Module):

    def __init__(self, d_x, d_a):
        super().__init__()
        self.query = nn.Linear(d_x, d_a, bias=False)
        self.key = nn.Linear(d_x, d_a, bias=False)
        self.value = nn.Linear(d_x, d_a, bias=False)
        self.output = nn.Linear(d_a, d_x, bias=False)
        for layer in (self.query, self.key):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)

    def forward(self, x):
        return (self.query(x), self.key(x), self.value(x))

class FullAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_x, self.d_s, self.d_a = args.gene_dim, args.pos_dim, args.gt_attention_dim
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('beta', torch.tensor(1.0))
        self.query_chunk = args.query_chunk
        self.register_buffer('radius', torch.tensor(float(args.gt_radius)))
        self.global_branch = AttentionBranch(self.d_x, self.d_a)
        self.local_branch = AttentionBranch(self.d_x, self.d_a)
        self.edge = nn.Sequential(nn.Linear(self.d_s, 2*self.d_a), nn.Softplus(),
            nn.Linear(2*self.d_a, 2*self.d_a), nn.Softplus(), nn.Linear(2*self.d_a, self.d_a))
        nn.init.xavier_uniform_(self.edge[-1].weight, gain=.01)
        nn.init.zeros_(self.edge[-1].bias)
        self.register_buffer('local_gate_tau', self.radius.new_tensor(float(args.local_mass_gate_tau)))
        self.exclude_local_from_global = True

    @torch.no_grad()
    def build_graph(self, s, receiver_ids=None, outer_radius=None):
        if not torch.isfinite(s).all():
            raise FloatingPointError('Nonfinite positions in radius graph')
        if s.shape[1] != 2:
            raise ValueError('Exact grid expects two spatial dimensions')
        ids = torch.arange(len(s), device=s.device) if receiver_ids is None else receiver_ids
        points = s.detach().double()
        outer = (self.radius.detach() if outer_radius is None else outer_radius).double() * (1 + 1e-07)
        if s.device.type == 'cpu':
            graph = []
            for block in ids.split(self.query_chunk):
                d = torch.cdist(points[block], points)
                (rows, cols) = torch.where((d <= outer) & (block[:, None] != torch.arange(len(s))[None, :]))
                graph.append((rows, cols))
            return graph
        grid = torch.floor((points - points.amin(0)) / outer).long()
        stride = grid[:, 1].amax() + 3
        keys = (grid[:, 0] + 1) * stride + (grid[:, 1] + 1)
        (sorted_keys, order) = torch.sort(keys)
        offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]], device=s.device, dtype=torch.long)
        graph = []
        for block in ids.split(self.query_chunk):
            cells = grid[block, None, :] + offsets[None, :, :] + 1
            query_keys = (cells[:, :, 0] * stride + cells[:, :, 1]).reshape(-1)
            lower = torch.searchsorted(sorted_keys, query_keys)
            upper = torch.searchsorted(sorted_keys, query_keys, right=True)
            lengths = upper - lower
            queries = torch.repeat_interleave(torch.arange(len(query_keys), device=s.device), lengths)
            prefix = lengths.cumsum(0) - lengths
            within = torch.arange(queries.numel(), device=s.device) - prefix[queries]
            cols = order[lower[queries] + within]
            rows = queries // 9
            distance = torch.linalg.vector_norm(points[block[rows]] - points[cols], dim=1)
            keep = (block[rows] != cols) & (distance <= outer)
            graph.append((rows[keep], cols[keep]))
        return graph

    @staticmethod
    def _segment_normalize(score, rows, n, valid):
        maximum = score.new_full((n,), -torch.inf)
        maximum.scatter_reduce_(0, rows, score.detach().masked_fill(~valid, -torch.inf), reduce='amax', include_self=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        shifted = torch.where(valid, score - maximum[rows], torch.zeros_like(score))
        weight = shifted.exp() * valid.to(score.dtype)
        denom = weight.new_zeros(n).index_add(0, rows, weight)
        safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        return (weight / safe[rows], maximum + safe.log(), denom > 0)

    def forward(self, x, s, ell, pool, return_components=False, graph=None, receiver_ids=None):
        if len(pool) < 2:
            raise ValueError('Global pool requires at least two senders')
        ids = torch.arange(len(x), device=x.device) if receiver_ids is None else receiver_ids
        (qg, kg, vg) = self.global_branch(x)
        (ql, kl, vl) = self.local_branch(x)
        if graph is None:
            graph = self.build_graph(s, ids)
        (gs, ls, counts, gates, supports, global_counts) = ([], [], [], [], [], [])
        for (block, (rows, cols)) in zip(ids.split(self.query_chunk), graph):
            allowed = block[:, None] != pool[None, :]
            distance = torch.linalg.vector_norm(s[block, None, :] - s[pool][None, :, :], dim=-1)
            allowed = allowed & (distance >= self.radius)
            score_g = qg[block] @ kg[pool].T / math.sqrt(self.d_a) + ell[pool][None, :]
            has_global = allowed.any(1)
            safe_scores = torch.where(has_global[:, None], score_g.masked_fill(~allowed, -torch.inf), torch.zeros_like(score_g))
            weights_g = torch.where(allowed, safe_scores.softmax(1), torch.zeros_like(score_g))
            gf = self.global_branch.output(weights_g @ vg[pool])
            count = ell.new_zeros(len(block))
            gate = ell.new_zeros(len(block))
            log_support = ell.new_full((len(block),), -torch.inf)
            lf = gf.new_zeros(len(block), self.d_x)
            if rows.numel():
                ds = s[cols] - s[block[rows]]
                valid = torch.linalg.vector_norm(ds, dim=1) < self.radius
                score_l = (ql[block[rows]] * (kl[cols] + self.edge(ds))).sum(1) / math.sqrt(self.d_a)
                (weights_l, _, _) = self._segment_normalize(score_l + ell[cols], rows, len(block), valid)
                (_, log_s, nonempty) = self._segment_normalize(ell[cols], rows, len(block), valid)
                gate = torch.where(nonempty, torch.sigmoid(log_s - self.local_gate_tau.log()), torch.zeros_like(log_s))
                log_support = torch.where(nonempty, log_s, torch.full_like(log_s, -torch.inf))
                aggregate = vl.new_zeros(len(block), self.d_a).index_add(0, rows, weights_l[:, None] * (vl[cols] - vl[block[rows]]))
                lf = gate[:, None] * self.local_branch.output(aggregate)
                count.index_add_(0, rows, valid.to(count.dtype))
            gs.append(gf)
            ls.append(lf)
            counts.append(count)
            gates.append(gate)
            supports.append(log_support)
            global_counts.append(allowed.sum(1))
        (gf, lf, count, gate, log_s, gcount) = map(torch.cat, (gs, ls, counts, gates, supports, global_counts))
        (gc, lc) = (self.alpha * gf, self.beta * lf)
        if return_components:
            return dict(total=gc + lc, global_field=gf, local_field=lf, global_contribution=gc, local_contribution=lc, alpha=self.alpha, beta=self.beta, neighbor_count=count, radius=self.radius, local_mass_gate=gate, local_log_total_mass=log_s, global_sender_count=gcount)
        return gc + lc

class PositionMLP(nn.Module):

    def __init__(self, d_x, d_s, hidden_dim=128, hidden_layers=6, time_scale=60.0):
        super().__init__()
        self.time_scale = float(time_scale)
        layers = []
        width = d_x + d_s + 1
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_dim), nn.ReLU()])
            width = hidden_dim
        layers.append(nn.Linear(width, d_s))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, x, s):
        time = torch.ones_like(s[:, :1]) * (t / self.time_scale)
        return self.net(torch.cat([x, s, time], dim=1))

class CellMassTurnoverDrift(nn.Module):
    """Signed proliferation/apoptosis rate for every particle.

    A positive output increases log mass and represents net proliferation. A
    negative output decreases log mass and represents net apoptosis. The
    virtual flag is an input feature, not a gate, so both real and virtual
    particles can move in either direction.
    """

    def __init__(self, gene_dim, pos_dim, rate_scale=0.1, time_scale=60.0):
        super().__init__()
        self.rate_scale = float(rate_scale)
        self.time_scale = max(float(time_scale), 1e-06)
        self.net = nn.Sequential(nn.Linear(gene_dim + pos_dim + 2, 64), nn.Softplus(), nn.Linear(64, 32), nn.Softplus(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, gene, pos, is_virtual):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=gene.dtype, device=gene.device)
        t = t.to(dtype=gene.dtype, device=gene.device)
        t_feature = torch.ones((gene.shape[0], 1), dtype=gene.dtype, device=gene.device) * (t.reshape(-1)[0] / self.time_scale)
        is_virtual = is_virtual.to(dtype=gene.dtype, device=gene.device)
        is_virtual = is_virtual.reshape(-1, 1)
        raw = self.net(torch.cat([gene, pos, t_feature, is_virtual], dim=1))
        return self.rate_scale * raw

def preserve_seed0_initialization(d_x, d_s):
    """Preserve draws of the archived temporary constructors, without storing them.

    The original model constructed and replaced attention and position modules.
    Their random draws affect the active growth network and training RNG stream.
    Constant layers/assignments consume no RNG and are omitted here.
    """
    q, k, v = [nn.Linear(d_x, d_x, bias=False) for _ in range(3)]
    for layer in (q, k):
        nn.init.xavier_uniform_(layer.weight, gain=.1)
    nn.init.xavier_uniform_(v.weight)
    first = nn.Linear(d_x, 4*d_x)
    last = nn.Linear(4*d_x, d_x)
    nn.init.xavier_uniform_(last.weight, gain=.01)
    # Archived temporary position constructor: rigid followed by local layers.
    for a, b in [(1, 16), (16, 16), (16, 3),
                 (d_x+d_s, 64), (64, 64), (64, d_s)]:
        nn.Linear(a, b)

