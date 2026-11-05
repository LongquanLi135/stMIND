"""Fixed manuscript settings: full-series axolotl and zebrafish, seed 0."""
from copy import deepcopy
from types import SimpleNamespace

FIT_TIMES = [2., 5., 10., 15., 20., 30., 60.]
SETTINGS = {'seed': 0,
 'fit_times': FIT_TIMES,
 'gene_dim': 10,
 'pos_dim': 2,
 'train_epochs': 3000,
 'train_lr': 0.001,
 'train_batch': 1500,
 'train_clip': 1,
 'train_dt': 1,
 'global_pool_size': 512,
 'query_chunk': 2048,
 'checkpoint_steps': True,
 'gt_attention_dim': 10,
 'gt_radius': 0.08222263582899947,
 'gt_initialization_seed': 0,
 'position_initialization_seed': 0,
 'position_hidden_dim': 128,
 'position_hidden_layers': 6,
 'position_time_scale': 60,
 'mass_dynamics_rate_scale': 0.1,
 'mass_dynamics_time_scale': 60,
 'mass_dynamics_energy_weight': 0.01,
 'sigma_const_gene': 0.01,
 'sigma_const_pos': 0,
 'local_mass_gate_tau': 418.0,
 'use_alpha': True,
 'use_robust': False,
 'alpha': 30,
 'sinkhorn_blur': 0.0005,
 'sinkhorn_scaling': 0.9,
 'reach': None,
 'lambda_marginal': 1,
 'lambda_energy': 1,
 'lambda_mass': 1,
 'lambda_shape': 0.15,
 'lambda_mass_dirichlet': 0.1,
 'lambda_mass_ess': 0.5,
 'mass_ess_target_ratio': 0.5,
 'mass_smooth_knn_k': 10,
 'shape_grid_size': 64,
 'shape_bandwidths': [0.02, 0.05, 0.1],
 'lambda_relative_growth': 10,
 'lambda_local_drift': 0.89,
 'lambda_orthogonality': 0.001,
 'orthogonality_warmup_epochs': 100,
 'orthogonality_ramp_epochs': 200,
 'lambda_local_mass': 0,
 'local_mass_warmup_epochs': 300,
 'local_mass_ramp_epochs': 300,
 'local_mass_target_chunk': 1024,
 'local_mass_prediction_chunk': 1024,
 'source_input_sha256': '4b6b21e2a6f48a8526e2b320a6d423ca59f683c6008e4b22093f644931d5a176',
 'selection_after_epoch': 600,
 'expected_stage_counts': [7668, 8106, 9436, 9663, 11316, 9252, 10964]}


DATASETS = ("axolotl", "zebrafish")

# Full-series coefficient-one seed0 experiment: zebrafish_3p3_24_full_seed0.
ZEBRAFISH = {'fit_times': [3.3, 5.25, 10.0, 12.0, 18.0, 24.0],
 'train_lr': 0.0003,
 'train_batch': 1000,
 'train_dt': 0.1,
 'gt_radius': 0.13455785996859185,
 'position_time_scale': 24,
 'mass_dynamics_rate_scale': 0.5,
 'mass_dynamics_time_scale': 18,
 'local_mass_gate_tau': 280.0,
 'use_alpha': False,
 'alpha': 100,
 'sinkhorn_blur': 0.03,
 'lambda_mass': 1.5,
 'lambda_shape': 0.1,
 'mass_ess_target_ratio': 0.35,
 'shape_grid_size': 24,
 'shape_bandwidths': [0.15, 0.35, 0.7],
 'source_input_sha256': 'c1eafae0c212ebeb25d61546b09297676d26e32bc35b95a741a55fb6b2a52b6d',
 'beta': 0.83,
 'expected_stage_counts': [1167, 562, 1036, 2081, 3048, 5257]}


def config(dataset="axolotl"):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    values = deepcopy(SETTINGS)
    if dataset == "zebrafish":
        values.update(deepcopy(ZEBRAFISH))
    values.update(dataset=dataset, time_unit="DPI" if dataset == "axolotl" else "hpf")
    return SimpleNamespace(**values)
