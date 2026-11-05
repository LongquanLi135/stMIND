"""Shared output, model construction and numerical checks."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda : handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)

def atomic_save(path, value):
    import torch
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    temp.replace(path)

def source_hashes():
    files = list(HERE.glob('*.py')) + list((HERE/'Eval').glob('*.py')) + list((HERE/'data').rglob('*.py'))
    return {str(p.relative_to(HERE)): digest(p) for p in sorted(files)}

def check_states(states, model):
    import torch
    mass = model.mass_from_log_mass(states[..., model.log_mass_index])
    if not torch.isfinite(states).all() or not torch.isfinite(mass).all() or (not (mass > 0).all()):
        raise FloatingPointError('Nonfinite state or mass overflow/underflow; no clipping applied')

def construct(args, device):
    from model import STMind
    model = STMind(args).to(device)
    if float(model.interaction.alpha) != 1.0 or float(model.interaction.beta) != 1.0:
        raise ValueError('Interaction coefficients must be 1/1')
    return model
