"""Load and validate the selected full-series PCA10 input."""
from pathlib import Path
from utils import digest


def load_data(path, args, device, *, smoke=False, allow_reprocessed=False):
    import numpy as np
    import torch
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data['x'], dtype=np.float32)
        s = np.asarray(data['s'], dtype=np.float32)
        times = np.asarray(data['times'], dtype=float)
    n = len(x)
    if x.shape != (n, 10) or s.shape != (n, 2) or times.shape != (n,):
        raise ValueError('Expected PCA10 expression, 2D positions and one time per cell')
    if not all((np.isfinite(a).all() for a in (x, s, times))):
        raise ValueError('Nonfinite input')
    if np.unique(times).tolist() != args.fit_times:
        raise ValueError(f"Expected {args.dataset} fitting times {args.fit_times} {args.time_unit}")
    counts = [int(np.count_nonzero(times == t)) for t in args.fit_times]
    if min(counts) < 2:
        raise ValueError('Each stage requires at least two cells')
    actual = digest(path)
    matches = actual == args.source_input_sha256
    if not smoke:
        if counts != args.expected_stage_counts:
            raise ValueError('Cell counts differ from the manuscript input')
        if not matches and (not allow_reprocessed):
            raise ValueError('Input SHA256 differs from the frozen manuscript input. Use --allow-reprocessed-input explicitly for a regenerated input.')
    rows = [dict(time=t, x=torch.as_tensor(x[times == t], device=device), s=torch.as_tensor(s[times == t], device=device)) for t in args.fit_times]
    return (rows, dict(path=str(path.resolve()), sha256=actual, reference_input_matches=matches, stage_counts=counts))
