"""Export the fitted full-series trajectory; no held-out forecasting task."""
import argparse
from pathlib import Path

from utils import atomic_json, check_states, construct, digest, source_hashes
from config import DATASETS, config
from load_Data import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="axolotl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-reprocessed-input", action="store_true")
    opt = parser.parse_args()
    import numpy as np
    import torch
    device = torch.device(opt.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("Use CPU or CUDA")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    args = config(opt.dataset)
    # Load the selected checkpoint produced here or the original trusted
    # manuscript checkpoint; its state-dict keys have been preserved.
    cp = torch.load(opt.checkpoint, map_location="cpu", weights_only=True)
    if cp["args"]["seed"] != 0 or cp["args"]["fit_times"] != args.fit_times:
        raise ValueError("Checkpoint is not the selected full-series seed0 experiment")
    rows, input_info = load_data(opt.input, args, device, smoke=bool(cp.get("smoke")),
                                allow_reprocessed=opt.allow_reprocessed_input)
    model = construct(args, device).eval()
    model.load_state_dict(cp["model_state_dict"], strict=True)
    if float(model.interaction.alpha) != 1. or float(model.interaction.beta) != 1.:
        raise ValueError("Checkpoint interaction coefficients are not 1/1")
    if not torch.isclose(model.interaction.radius, model.interaction.radius.new_tensor(args.gt_radius)) or not torch.isclose(model.interaction.local_gate_tau, model.interaction.local_gate_tau.new_tensor(args.local_mass_gate_tau)):
        raise ValueError("Checkpoint spatial calibration differs from the manuscript")
    initial = model.initial_state(rows[0])
    generator = torch.Generator(device=device).manual_seed(4900600)
    pool = torch.randperm(len(initial), device=device, generator=generator)[:args.global_pool_size]
    opt.output.mkdir(parents=True, exist_ok=False)
    files = {}
    with torch.no_grad():
        states = model.rollout(args.fit_times, initial, pool, brownian_seed=4900600, checkpoint_steps=False)
        check_states(states, model)
        for t, state in zip(args.fit_times, states):
            fields = model.drift_components(state, pool)
            x, s = state[:, :model.d_x], state[:, model.d_x:model.d]
            ell = state[:, model.log_mass_index]
            arrays = {k: v.cpu().numpy() for k, v in fields.items() if torch.is_tensor(v)}
            arrays.update(x=x.cpu().numpy(), s=s.cpu().numpy(), mass=ell.exp().cpu().numpy(),
                          log_mass=ell.cpu().numpy(),
                          position_drift=model.position(state.new_tensor(t), x, s).cpu().numpy(),
                          growth_rate=model.growth(state.new_tensor(t), x, s, state.new_zeros(len(x), 1)).cpu().numpy())
            path = opt.output / f"fit_{t:g}.npz"
            np.savez_compressed(path, **arrays)
            files[path.name] = digest(path)
    atomic_json(opt.output / "ready.json", dict(status="finished", task_scope="fitted_reconstruction_not_forecast_test",
        dataset=args.dataset, fit_times=args.fit_times, checkpoint_sha256=digest(opt.checkpoint), checkpoint_epoch=cp["epoch"],
        input=input_info, seed=0, brownian_seed=4900600, global_pool_seed=4900600,
        global_pool_ids=pool.cpu().tolist(), files_sha256=files, smoke=bool(cp.get("smoke")),
        device=str(device), source_sha256=source_hashes()))


if __name__ == "__main__":
    main()
