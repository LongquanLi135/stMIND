"""Train the full axolotl or zebrafish model with seed 0."""
import argparse
import json
import random
import time
from pathlib import Path

from utils import atomic_json, atomic_save, check_states, construct, source_hashes
from config import DATASETS, config
from load_Data import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="axolotl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="CUDA for the full experiment; CPU for small smoke checks")
    parser.add_argument("--smoke", action="store_true", help="Four updates; explicitly marked as a software check")
    parser.add_argument("--allow-reprocessed-input", action="store_true")
    opt = parser.parse_args()
    import numpy as np
    import torch
    from losses import endpoint_loss, path_energy, sample_endpoints
    from losses import local_drift_path
    from losses import relative_growth_path
    from losses import endpoint_orthogonality
    from losses import _make_spatial_grid

    device = torch.device(opt.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("Use a CPU or CUDA device")
    args = config(opt.dataset)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    rows, input_info = load_data(opt.input, args, device, smoke=opt.smoke,
                                allow_reprocessed=opt.allow_reprocessed_input)
    # Initialization is deliberately not enclosed in fork_rng: preserve the
    # archived seed0 stream before sampling the first pool and endpoint batches.
    model = construct(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.9)
    initial = model.initial_state(rows[0])
    points = torch.cat([r["s"] for r in rows])
    grid = _make_spatial_grid(points.amin(0), points.amax(0), args.shape_grid_size,
                              device, initial.dtype)
    epochs = 4 if opt.smoke else args.train_epochs
    opt.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    atomic_json(opt.output / "config.json", vars(args))
    status = dict(status="running", dataset=args.dataset, seed=0, variant="full", fit_times=args.fit_times,
                  task_scope="software_smoke_check" if opt.smoke else "full_series_fitting",
                  independent_forecast_test=False, input=input_info,
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  device=str(device), source_sha256=hashes, total_epochs=epochs, epoch=0)
    atomic_json(opt.output / "status.json", status)
    best = late_best = float("inf")
    started = time.monotonic()
    try:
        with (opt.output / "metrics.jsonl").open("x") as log:
            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                pool = torch.randperm(len(initial), device=device)[:min(args.global_pool_size, len(initial))]
                plan = sample_endpoints(rows, args.train_batch)
                states = model.rollout(args.fit_times, initial, pool, brownian_seed=26000000 + epoch)
                check_states(states, model)
                loss_epoch = (1, 200, 301, 600)[epoch - 1] if opt.smoke else epoch
                endpoint, parts = endpoint_loss(model, states, rows, args, grid, plan, loss_epoch)
                energy = path_energy(states, model, args.fit_times)
                relative = relative_growth_path(states, model, args.fit_times)
                local = local_drift_path(states, model, args.fit_times)
                orth, orth_raw, orth_weight, _ = endpoint_orthogonality(model, states, pool, args, loss_epoch)
                # Keep the archived arithmetic order as well as all weights.
                loss = endpoint + args.lambda_energy * energy + orth + args.lambda_relative_growth * relative + args.lambda_local_drift * local
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_clip, error_if_nonfinite=True)
                score = float(torch.stack([p["base_objective"] for p in parts]).mean().detach())
                selected = dict(model_state_dict=model.state_dict(), args=vars(args), epoch=epoch,
                                loss=score, weights_timing="pre_update_matches_scored_trajectory",
                                source_sha256=hashes, input=input_info, smoke=opt.smoke)
                if score < best:
                    best = score
                    atomic_save(opt.output / "train.best.pt", selected)
                if loss_epoch >= args.selection_after_epoch and score < late_best:
                    late_best = score
                    atomic_save(opt.output / "train.best_localmass.pt", selected)
                optimizer.step()
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                record = dict(epoch=epoch, loss=float(loss.detach()), selection_loss=score,
                              energy=float(energy.detach()), relative_growth=float(relative.detach()),
                              local_drift=float(local.detach()), orthogonality=float(orth_raw.detach()),
                              orthogonality_weight=orth_weight, gradient_norm=float(grad),
                              elapsed_seconds=time.monotonic() - started)
                log.write(json.dumps(record) + "\n")
                log.flush()
                atomic_save(opt.output / "train.last.pt", dict(
                    model_state_dict=model.state_dict(), args=vars(args), epoch=epoch,
                    best=best, best_localmass=None if late_best == float("inf") else late_best,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rng_state=dict(torch=torch.get_rng_state(),
                                   cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else []),
                    source_sha256=hashes, input=input_info, smoke=opt.smoke,
                    weights_timing="post_update"))
                status.update(epoch=epoch, latest=record)
                atomic_json(opt.output / "status.json", status)
                if epoch <= 4 or epoch % 100 == 0:
                    print(json.dumps(record), flush=True)
        status["status"] = "finished"
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        atomic_json(opt.output / "status.json", status)


if __name__ == "__main__":
    main()
