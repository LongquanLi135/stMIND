"""Losses used by the full axolotl and zebrafish experiments."""

import math
import torch
from geomloss import SamplesLoss

def calc_gene_ot_loss(x_pred, x_target, p=None, q=None, blur=0.05, scaling=0.9, reach=None):
    n_source = x_pred.shape[0]
    n_target = x_target.shape[0]
    if p is None:
        p = torch.ones(n_source, dtype=x_pred.dtype, device=x_pred.device)
        p = p / n_source
    if q is None:
        q = torch.ones(n_target, dtype=x_target.dtype, device=x_target.device)
        q = q / n_target
    p = p.contiguous().view(-1)
    q = q.contiguous().view(-1)
    loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=blur, scaling=scaling, reach=reach)
    return loss_fn(p, x_pred.contiguous(), q, x_target.contiguous())

def calc_joint_ot_loss(pred_gene, pred_pos, true_gene, true_pos, p_source, q_target, alpha, blur, scaling, reach):
    joint_pred = torch.cat([pred_gene, pred_pos * alpha], dim=1)
    joint_true = torch.cat([true_gene, true_pos * alpha], dim=1)
    return calc_gene_ot_loss(joint_pred, joint_true, p=p_source, q=q_target, blur=blur, scaling=scaling, reach=reach)

def calc_joint_ot_loss_beta(pred_gene, pred_pos, true_gene, true_pos, p_source, q_target, beta, blur, scaling, reach):
    gene_weight = math.sqrt(1.0 - beta)
    pos_weight = math.sqrt(beta)
    joint_pred = torch.cat([pred_gene * gene_weight, pred_pos * pos_weight], dim=1)
    joint_true = torch.cat([true_gene * gene_weight, true_pos * pos_weight], dim=1)
    return calc_gene_ot_loss(joint_pred, joint_true, p=p_source, q=q_target, blur=blur, scaling=scaling, reach=reach)

def _make_spatial_grid(pos_min, pos_max, grid_size, device, dtype):
    """Build one fixed grid shared by every time point and transition."""
    if grid_size < 4:
        raise ValueError('shape_grid_size must be at least 4')
    span = (pos_max - pos_min).clamp_min(1e-06)
    lower = pos_min - 0.05 * span
    upper = pos_max + 0.05 * span
    axis_x = torch.linspace(lower[0], upper[0], grid_size, device=device, dtype=dtype)
    axis_y = torch.linspace(lower[1], upper[1], grid_size, device=device, dtype=dtype)
    (grid_x, grid_y) = torch.meshgrid(axis_x, axis_y, indexing='ij')
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

def _normalized_kde(pos, weights, grid, bandwidth):
    """Evaluate a mass-normalized Gaussian KDE on a fixed two-dimensional grid."""
    if bandwidth <= 0:
        raise ValueError('shape bandwidths must be positive')
    weights = weights.contiguous().view(-1).clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-08)
    squared_dist = torch.sum((grid[:, None, :] - pos[None, :, :]) ** 2, dim=-1)
    kernel = torch.exp(-0.5 * squared_dist / bandwidth ** 2)
    density = torch.matmul(kernel, weights)
    return density / density.sum().clamp_min(1e-08)

def calc_multiscale_shape_loss(pred_pos, target_pos, pred_mass, target_mass, grid, bandwidths):
    """Compare normalized spatial morphology independently of total cell mass."""
    losses = []
    for bandwidth in bandwidths:
        pred_density = _normalized_kde(pred_pos, pred_mass, grid, float(bandwidth))
        target_density = _normalized_kde(target_pos, target_mass, grid, float(bandwidth))
        loss = 0.5 * torch.sum((torch.sqrt(pred_density + 1e-08) - torch.sqrt(target_density + 1e-08)) ** 2)
        losses.append(loss)
    return torch.stack(losses).mean()

def calc_mass_rate_dirichlet(rate, pos, k):
    """Smooth signed proliferation/apoptosis rates on a spatial KNN graph."""
    n_particles = pos.shape[0]
    if n_particles < 2:
        return torch.zeros((), dtype=pos.dtype, device=pos.device)
    k = min(max(int(k), 1), n_particles - 1)
    with torch.no_grad():
        distance = torch.cdist(pos, pos)
        distance.fill_diagonal_(float('inf'))
        (neighbor_distance, neighbor_index) = torch.topk(distance, k=k, dim=1, largest=False)
        scale = neighbor_distance[:, -1].median().clamp_min(1e-06)
        edge_weight = torch.exp(-0.5 * (neighbor_distance / scale) ** 2)
    neighbor_rate = rate.view(-1)[neighbor_index]
    rate_difference = rate.view(-1, 1) - neighbor_rate
    return torch.sum(edge_weight * rate_difference ** 2) / edge_weight.sum().clamp_min(1e-08)

def calc_mass_ess_floor(particle_mass, target_ratio):
    """Penalize only excessive mass concentration, not biological heterogeneity."""
    normalized_mass = particle_mass / particle_mass.sum().clamp_min(1e-08)
    ess = 1.0 / torch.sum(normalized_mass ** 2).clamp_min(1e-08)
    ess_ratio = ess / float(particle_mass.numel())
    loss = torch.relu(torch.as_tensor(target_ratio, dtype=particle_mass.dtype, device=particle_mass.device) - ess_ratio) ** 2
    return (loss, ess_ratio)

def _joint_loss(pred_gene, pred_pos, target_gene, target_pos, p, q, args):
    if args.use_robust:
        raise ValueError("The selected experiments do not use robust rematching")
    if args.use_alpha:
        return calc_joint_ot_loss(pred_gene, pred_pos, target_gene, target_pos, p, q,
                                  args.alpha, args.sinkhorn_blur, args.sinkhorn_scaling, args.reach)
    return calc_joint_ot_loss_beta(pred_gene, pred_pos, target_gene, target_pos, p, q,
                                  args.beta, args.sinkhorn_blur, args.sinkhorn_scaling, args.reach)

def local_mass_weight(epoch, args):
    progress = (epoch - args.local_mass_warmup_epochs) / args.local_mass_ramp_epochs
    return args.lambda_local_mass * min(1.0, max(0.0, progress))

@torch.no_grad()
def expression_nearest_counts(predicted_x, target_x, target_chunk=1024, prediction_chunk=1024):
    if min(len(predicted_x), len(target_x), target_chunk, prediction_chunk) < 1:
        raise ValueError('Populations and block sizes must be positive')
    counts = predicted_x.new_zeros(len(predicted_x))
    for target in target_x.split(target_chunk):
        best = target.new_full((len(target),), torch.inf)
        index = torch.zeros(len(target), device=target.device, dtype=torch.long)
        for start in range(0, len(predicted_x), prediction_chunk):
            distance = torch.cdist(target, predicted_x[start:start + prediction_chunk], compute_mode='donot_use_mm_for_euclid_dist')
            (value, local) = distance.min(1)
            replace = value < best
            index = torch.where(replace, local + start, index)
            best = torch.minimum(best, value)
        counts.index_add_(0, index, torch.ones_like(best))
    return counts

def normalized_local_mass_loss(predicted_x, target_x, mass, args, elapsed_time):
    counts = expression_nearest_counts(predicted_x, target_x, args.local_mass_target_chunk, args.local_mass_prediction_chunk)
    mean_target_mass = len(target_x) / len(predicted_x)
    residual = (mass - counts) / mean_target_mass
    loss = residual.square().mean()
    q = counts / len(target_x)
    target_ess_ratio = 1.0 / (len(predicted_x) * q.square().sum())
    diagnostics = dict(local_mass=loss, local_mass_count_sum=counts.sum(), local_mass_max_count=counts.max(), local_mass_empty_fraction=(counts == 0).float().mean(), local_mass_target_ess_ratio=target_ess_ratio, local_mass_target_below_ess_floor=target_ess_ratio < args.mass_ess_target_ratio, local_mass_max_reachable=None, local_mass_target_fraction_above_reachable=None, local_mass_target_cell_fraction_above_reachable=None, local_mass_mean_target=mean_target_mass)
    return (loss, diagnostics)

def path_energy(states, model, times):
    initial_mass = model.mass_from_log_mass(states[0, :, model.log_mass_index]).sum()
    action = (states[-1, :, model.energy_index] - states[0, :, model.energy_index]).sum()
    return action / (initial_mass * (float(times[-1]) - float(times[0])))

def endpoint_loss(model, states, rows, args, grid, sample_plan, epoch):
    n0 = len(rows[0]['x'])
    items = []
    for (j, target) in enumerate(rows[1:], 1):
        (pred_ids, target_ids) = sample_plan[j - 1]
        current = states[j]
        sample = current[pred_ids]
        (x, s) = (sample[:, :model.d_x], sample[:, model.d_x:model.d])
        mass_all = model.mass_from_log_mass(current[:, model.log_mass_index])
        mass = mass_all[pred_ids]
        ratio = len(target['x']) / n0
        p = mass / len(pred_ids)
        q = mass.new_full((len(target_ids),), ratio / len(target_ids))
        (tx, ts) = (target['x'][target_ids], target['s'][target_ids])
        if args.reach is not None:
            raise ValueError('v41 matching requires reach=None and unit-total weights')
        p_match = mass / mass.sum()
        q_match = torch.ones_like(q) / len(q)
        ot = _joint_loss(x, s, tx, ts, p_match, q_match, args)
        predicted_ratio = mass_all.sum() / n0
        mass_loss = (torch.logsumexp(current[:, model.log_mass_index], dim=0) - current.new_tensor(float(len(target['x']))).log()).square()
        shape = calc_multiscale_shape_loss(s, ts, mass, q, grid, args.shape_bandwidths)
        rate = model.growth(s.new_tensor(target['time']), x, s, s.new_zeros(len(s), 1)).view(-1)
        rate_dir = calc_mass_rate_dirichlet(rate, s, args.mass_smooth_knn_k)
        (ess_penalty, ess_ratio) = calc_mass_ess_floor(mass_all, args.mass_ess_target_ratio)
        base_objective = args.lambda_marginal * ot + args.lambda_mass * mass_loss + args.lambda_shape * shape + args.lambda_mass_dirichlet * rate_dir + args.lambda_mass_ess * ess_penalty
        (lm, lm_diagnostic) = normalized_local_mass_loss(current[:, :model.d_x], target['x'], mass_all, args, target['time'] - rows[0]['time'])
        lm_weight = local_mass_weight(epoch, args)
        objective = base_objective + lm_weight * lm
        items.append(dict(time=target['time'], objective=objective, ot=ot, mass=mass_loss, shape=shape, rate_dirichlet=rate_dir, ess_penalty=ess_penalty, ess_ratio=ess_ratio, predicted_mass=mass_all.sum(), target_mass=len(target['x']), source_mass_ratio_estimate=p.sum(), target_mass_ratio=q.sum(), matching_source_mass=p_match.sum(), matching_target_mass=q_match.sum(), base_objective=base_objective, local_mass_weight=lm_weight, local_mass_weighted=lm_weight * lm, **lm_diagnostic))
    return (torch.stack([r['objective'] for r in items]).mean(), items)

def sample_endpoints(rows, batch):
    device = rows[0]['x'].device
    n = len(rows[0]['x'])
    return [(torch.randperm(n, device=device)[:min(batch, n)], torch.randperm(len(r['x']), device=device)[:min(batch, len(r['x']))]) for r in rows[1:]]

def orthogonality_weight(epoch, args):
    return args.lambda_orthogonality * min(1.0, max(0.0, (epoch - args.orthogonality_warmup_epochs) / args.orthogonality_ramp_epochs))

def component_penalty(intrinsic, global_contribution, local_contribution, ell):
    p = ell.detach().softmax(0)
    return (p * ((intrinsic * global_contribution).sum(-1).abs() + (intrinsic * local_contribution).sum(-1).abs())).sum()

def endpoint_orthogonality(model, states, pool, args, epoch):
    weight = orthogonality_weight(epoch, args)
    (values, diagnostics) = ([], [])
    with torch.set_grad_enabled(torch.is_grad_enabled() and weight > 0):
        for state in states[1:]:
            fields = model.drift_components(state, pool)
            (intra, global_, local) = (fields['intrinsic'], fields['global_contribution'], fields['local_contribution'])
            ell = state[:, model.log_mass_index]
            p = ell.detach().softmax(0)
            raw = component_penalty(intra, global_, local, ell)
            values.append(raw)

            def rms(v):
                return float((p * v.detach().square().sum(-1)).sum().sqrt())

            def cos2(a, b):
                (a, b) = (a.detach(), b.detach())
                return float((p * (a * b).sum(-1).square() / ((a.square().sum(-1) + 1e-08) * (b.square().sum(-1) + 1e-08))).sum())
            diagnostics.append(dict(orthogonality=float(raw.detach()), absdot_self_global=float((p * (intra.detach() * global_.detach()).sum(-1).abs()).sum()), absdot_self_local=float((p * (intra.detach() * local.detach()).sum(-1).abs()).sum()), rms_intrinsic=rms(intra), rms_global_contribution=rms(global_), rms_local_contribution=rms(local), rms_global_raw=rms(fields['global_field']), rms_local_raw=rms(fields['local_field']), rms_total=rms(fields['expression_total']), cosine2_self_global=cos2(intra, global_), cosine2_self_local=cos2(intra, local), cosine2_global_local=cos2(global_, local), mean_neighbor_count=float(fields['neighbor_count'].mean()), empty_neighbor_fraction=float((fields['neighbor_count'] == 0).float().mean())))
    raw = torch.stack(values).mean()
    return (weight * raw, raw, weight, diagnostics)

def centered_growth_variance(rate, log_mass):
    rate = rate.reshape(-1)
    p = log_mass.reshape(-1).detach().softmax(0)
    mean = (p * rate).sum()
    return (p * (rate - mean).square()).sum()

def relative_growth_path(states, model, times):
    integral = (states[-1, :, model.d + 2] - states[0, :, model.d + 2]).mean()
    return integral / (float(times[-1]) - float(times[0]))

def local_drift_penalty(local_contribution, log_mass):
    p = log_mass.reshape(-1).detach().softmax(0)
    return (p * local_contribution.square().sum(-1)).sum()

def local_drift_path(states, model, times):
    integral = (states[-1, :, model.d + 3] - states[0, :, model.d + 3]).mean()
    return integral / (float(times[-1]) - float(times[0]))
