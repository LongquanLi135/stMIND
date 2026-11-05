"""Read fitted fields, sender-type local contributions and nonlocal attention."""
import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import atomic_json, construct, digest, source_hashes
from config import DATASETS, config
from load_Data import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="axolotl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True, help="analysis_inputs produced by prepare.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-reprocessed-input", action="store_true")
    opt = parser.parse_args()
    import numpy as np
    import torch
    # The manuscript observed-cell readout used CPU pools. Keep that stream
    # separate from the CUDA pool used by a fitted trajectory rollout.
    torch.set_num_threads(2)
    torch.manual_seed(0)
    args = config(opt.dataset)
    cp = torch.load(opt.checkpoint, map_location="cpu", weights_only=True)
    if cp["args"]["seed"] != 0 or cp["args"]["fit_times"] != args.fit_times:
        raise ValueError("Checkpoint is not the selected full-series seed0 run")
    rows, input_info = load_data(opt.input, args, "cpu", smoke=bool(cp.get("smoke")),
                                allow_reprocessed=opt.allow_reprocessed_input)
    model = construct(args, "cpu").eval().requires_grad_(False)
    model.load_state_dict(cp["model_state_dict"], strict=True)
    branch = model.interaction
    if float(branch.alpha) != 1. or float(branch.beta) != 1.:
        raise ValueError("Checkpoint interaction coefficients are not 1/1")
    if not torch.isclose(branch.radius, branch.radius.new_tensor(args.gt_radius)) or not torch.isclose(branch.local_gate_tau, branch.local_gate_tau.new_tensor(args.local_mass_gate_tau)):
        raise ValueError("Checkpoint spatial calibration differs from the manuscript")
    branch.query_chunk = 128
    raw = []
    for row in rows:
        with np.load(opt.observed / f"observed_{row['time']:g}.npz", allow_pickle=False) as data:
            d = {k: data[k] for k in ("x", "s", "ids", "labels")}
        np.testing.assert_array_equal(d["x"], row["x"].numpy())
        np.testing.assert_array_equal(d["s"], row["s"].numpy())
        raw.append(d)
    types = np.unique(np.concatenate([d["labels"] for d in raw]))
    nt = len(types)
    opt.output.mkdir(parents=True, exist_ok=False)
    records = []
    with torch.no_grad():
        for row, d in zip(rows, raw):
            t, x, s = row["time"], row["x"], row["s"]
            n = len(x)
            state = model.initial_state(row)
            ell = state[:, model.log_mass_index]
            pool_seed = 4900600 + round(t * 100)
            pool = torch.randperm(n, generator=torch.Generator().manual_seed(pool_seed))[:args.global_pool_size]
            fields = model.drift_components(state, pool)
            qg, kg, vg = branch.global_branch(x)
            weights = torch.empty(n, len(pool))
            for block in torch.arange(n).split(128):
                allowed = (block[:, None] != pool[None, :]) & (torch.linalg.vector_norm(s[block, None, :] - s[pool][None, :, :], dim=-1) >= branch.radius)
                scores = qg[block] @ kg[pool].T / math.sqrt(branch.d_a) + ell[pool][None, :]
                safe = torch.where(allowed.any(1)[:, None], scores.masked_fill(~allowed, -torch.inf), torch.zeros_like(scores))
                weights[block] = torch.where(allowed, safe.softmax(1), torch.zeros_like(scores))
            restored = branch.alpha * branch.global_branch.output(weights @ vg[pool])
            torch.testing.assert_close(restored, fields["global_contribution"], rtol=2e-5, atol=2e-6)
            ql, kl, vl = branch.local_branch(x)
            type_ids = torch.tensor(np.searchsorted(types, d["labels"]))
            by_type = torch.zeros(n, nt, model.d_x)
            attention = torch.zeros(n, nt)
            availability = torch.zeros(n, nt)
            graph = branch.build_graph(s)
            for block, (rr, cc) in zip(torch.arange(n).split(branch.query_chunk), graph):
                if not len(rr):
                    continue
                ds = s[cc] - s[block[rr]]
                valid = torch.linalg.vector_norm(ds, dim=1) < branch.radius
                scores = (ql[block[rr]] * (kl[cc] + branch.edge(ds))).sum(1) / math.sqrt(branch.d_a) + ell[cc]
                w, _, _ = branch._segment_normalize(scores, rr, len(block), valid)
                messages = branch.local_branch.output(vl[cc] - vl[block[rr]])
                messages = branch.beta * fields["local_mass_gate"][block[rr], None] * w[:, None] * messages
                flat = rr * nt + type_ids[cc]
                by_type[block] = torch.zeros(len(block) * nt, model.d_x).index_add_(0, flat, messages).reshape(len(block), nt, model.d_x)
                attention[block] = torch.zeros(len(block) * nt).index_add_(0, flat, w).reshape(len(block), nt)
                availability[block] = torch.zeros(len(block) * nt).index_add_(0, flat, valid.float()).reshape(len(block), nt)
            torch.testing.assert_close(by_type.sum(1), fields["local_contribution"], rtol=3e-4, atol=3e-5)
            torch.testing.assert_close(availability.sum(1), fields["neighbor_count"])
            arrays = {k: fields[k].numpy() for k in ("intrinsic", "global_contribution", "local_contribution", "expression_total", "neighbor_count", "local_mass_gate", "global_sender_count")}
            arrays.update(x=x.numpy(), s=s.numpy(), ids=d["ids"], labels=d["labels"], pool=pool.numpy(),
                          sender_types=types, local_by_sender_type=by_type.numpy(),
                          attention_by_sender_type=attention.numpy(), neighbor_counts_by_sender_type=availability.numpy(),
                          nonlocal_attention=weights.numpy(),
                          position_drift=model.position(state.new_tensor(t), x, s).numpy())
            if not all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in "fc"):
                raise FloatingPointError("Nonfinite observed readout")
            path = opt.output / f"observed_{t:g}.npz"
            np.savez_compressed(path, **arrays)
            records.append(dict(time=t, n_cells=n, pool_seed=pool_seed, file=path.name,
                                sha256=digest(path), max_sender_decomposition_error=float((by_type.sum(1) - fields["local_contribution"]).abs().max())))
    atomic_json(opt.output / "manifest.json", dict(status="finished", dataset=args.dataset, seed=0, fit_times=args.fit_times,
        checkpoint_sha256=digest(opt.checkpoint), checkpoint_epoch=cp["epoch"], input=input_info,
        protocol="Observed training cells, unit masses, complete local radius graph, fixed CPU sender pool per stage",
        nonlocal_matrix="Rows are focal cells; columns follow pool; masked entries are zero",
        records=records, smoke=bool(cp.get("smoke")), source_sha256=source_hashes()))


if __name__ == "__main__":
    main()
