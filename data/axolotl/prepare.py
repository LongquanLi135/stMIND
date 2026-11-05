"""Prepare the original full-series axolotl counts protocol, seven stages, seed 0."""
import os
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMBA_NUM_THREADS='4')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import joblib
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

def spatial_scales(rows):
    n0 = len(rows[0]['x'])
    points = [r['s'].detach().cpu().double().numpy() for r in rows]
    positive = []
    for s in points:
        d = cKDTree(s).query(s, k=9, workers=1)[0][:, -1]
        positive.append(d[d > 0])
    reference = float(np.median(np.concatenate(positive)))
    radius = 2 * reference
    supports = []
    counts = []
    for s in points:
        tree = cKDTree(s)
        g = tree.sparse_distance_matrix(tree, radius, output_type='coo_matrix')
        keep = g.row != g.col
        recv = g.row[keep]
        u = np.minimum(g.data[keep] / radius, 1.0)
        z = np.bincount(recv, weights=(1 - u) ** 4 * (1 + 4 * u), minlength=len(s)) / n0
        supports.append(z[z > 0])
        counts.append(np.bincount(recv, minlength=len(s)))
    tau = float(0.1 * np.median(np.concatenate(supports)))
    if not np.isfinite(reference + tau) or min(reference, tau) <= 0:
        raise ValueError('Invalid radius scale')
    return dict(reference_radius=reference, support_tau=tau, n_reference=n0, initial_radius=radius, radius_bounds=[reference, 4 * reference], nonself_k_reference=8, reference_neighbor_quantiles=np.quantile(np.concatenate(counts), [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist())
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FIT_TIMES
from utils import digest, atomic_json

BATCHES = {2: 'Injury_2DPI_rep1_SS200000147BL_D5', 5: 'Injury_5DPI_rep1_SS200000147BL_D2', 10: 'Injury_10DPI_rep1_SS200000147BL_B5', 15: 'Injury_15DPI_rep4_FP200000266TR_E4', 20: 'Injury_20DPI_rep2_SS200000147BL_B4', 30: 'Injury_30DPI_rep2_FP200000264BL_A6', 60: 'Injury_60DPI_rep3_FP200000264BL_A6'}

def normalize_once(counts):
    c = sparse.csr_matrix(counts, dtype=np.float64)
    totals = np.asarray(c.sum(1)).ravel()
    assert np.all(totals > 0)
    y = (sparse.diags(10000.0 / totals) @ c).tocsr()
    y.data = np.log1p(y.data)
    return y

def main():
    p = argparse.ArgumentParser(description='Prepare the manuscript axolotl 2--60 DPI input')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    opt = p.parse_args()
    case = 'axolotl'
    source = opt.source
    out = opt.output
    out.mkdir(parents=True, exist_ok=False)
    assert not (out / 'ready.json').exists(), 'Refusing to overwrite frozen full-series inputs'
    print('READING_COUNTS', case, source, flush=True)
    raw = ad.read_h5ad(source)
    raw.obs_names_make_unique()
    assert raw.var_names.is_unique and 'counts' in raw.layers
    c = sparse.csr_matrix(raw.layers['counts'], dtype=np.float64)
    assert np.isfinite(c.data).all() and np.all(c.data >= 0)
    integer_error = float(np.max(np.abs(c.data - np.rint(c.data))))
    assert integer_error < 1e-05
    ids = np.asarray(raw.obs_names, dtype=str)
    genes = np.asarray(raw.var_names, dtype=str)
    pos = np.asarray(raw.obsm['spatial'], dtype=np.float64)
    assert np.isfinite(pos).all()
    fit = FIT_TIMES
    training_stats = {}
    batches = raw.obs['Batch'].astype(str).to_numpy()
    labels = raw.obs['Annotation'].astype(str).to_numpy(dtype=str)
    indices = []
    days = []
    coordinates = []
    for t in fit:
        ix = np.flatnonzero(batches == BATCHES[int(t)])
        assert len(ix) > 0
        n_before = len(ix)
        s0 = pos[ix]
        keep = np.ones(len(ix), dtype=bool)
        if t == 10:
            keep &= s0[:, 0] <= 28000
        if t == 15:
            keep &= ~((s0[:, 0] > 6050) | (s0[:, 1] < 8000) | (s0[:, 1] < 8500) & (s0[:, 0] < 3000))
        if t == 20:
            keep &= ~((s0[:, 1] < 7000) | (s0[:, 0] > 16000))
        ix = ix[keep]
        spatial_removed = n_before - len(ix)
        cell_keep = np.asarray((c[ix] > 0).sum(1)).ravel() >= 100
        ix = ix[cell_keep]
        s0 = pos[ix]
        assert len(ix) > 0
        lo = s0.min(0)
        span = float(s0[:, 0].max() - lo[0])
        assert span > 0
        indices.extend(ix)
        days.extend([t] * len(ix))
        coordinates.append((s0 - lo) / span)
        training_stats[str(t)] = dict(before=n_before, spatial_removed=spatial_removed, min_genes_removed=int((~cell_keep).sum()), retained=len(ix), spatial_min=lo.tolist(), spatial_x_span=span)
    ix = np.asarray(indices, dtype=int)
    tt = np.asarray(days, float)
    s = np.concatenate(coordinates).astype(np.float32)
    gene_keep = np.asarray((c[ix] > 0).sum(0)).ravel() >= 3
    retained_genes = genes[gene_keep]
    train_counts = c[ix][:, gene_keep]
    batches_fit = batches[ix]
    assert len(ix) == len(np.unique(ix)) and sorted(np.unique(tt)) == fit
    normalized = normalize_once(train_counts)
    direct = train_counts[:24].toarray()
    direct = np.log1p(10000.0 * direct / direct.sum(1, keepdims=True))
    np.testing.assert_allclose(normalized[:24].toarray(), direct, rtol=1e-12, atol=1e-12)
    ann = ad.AnnData(X=normalized, obs=pd.DataFrame({'Batch': batches_fit}, index=pd.Index(ids[ix])), var=pd.DataFrame(index=pd.Index(retained_genes)))
    sc.pp.highly_variable_genes(ann, n_top_genes=2000, batch_key='Batch', flavor='seurat', subset=False)
    hvg = ann.var.highly_variable.to_numpy(bool)
    hvg_genes = retained_genes[hvg]
    features = normalized[:, hvg].toarray()
    print('FITTING_FULLSERIES_PCA', case, len(ix), len(hvg_genes), flush=True)
    pca = PCA(n_components=10, svd_solver='arpack', random_state=0).fit(features)
    x = pca.transform(features).astype(np.float32)
    A = np.diag(1 / np.sqrt(pca.explained_variance_))
    b = np.zeros(10)
    z = x.astype(float) @ A
    transform = dict(pca=pca, retained_genes=retained_genes, hvg_genes=hvg_genes, gene_keep=gene_keep, hvg_mask=hvg, gene_scaler=None, pc_scaler=None)
    spatial_policy = 'per slice (s-axis minima)/x-span; original fixed QC and geometry rules'
    normalization = 'counts total10000 over full-training min-cells3 retained genes; log1p exactly once'
    assert pca.n_samples_ == len(ix) and x.shape == (len(ix), 10) and np.isfinite(x).all()
    assert np.isfinite(s).all() and np.isfinite(z).all()
    del ann, normalized, features, train_counts
    source_hash = digest(source)
    transform.update(fit_times=fit, train_ids=ids[ix], source_sha256=source_hash, normalization=normalization, task_scope='full_series_fitting')
    np.savez_compressed(out / 'training.npz', x=x, s=s, times=tt, cell_ids=ids[ix], genes=hvg_genes)
    joblib.dump(transform, out / 'transform.joblib')
    analysis = out / 'analysis_inputs'
    analysis.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(analysis / 'map.npz', A=A, b=b)
    for t in fit:
        q = tt == t
        np.savez_compressed(analysis / f'observed_{t:g}.npz', x=x[q], s=s[q], z=z[q], ids=ids[ix[q]], labels=labels[ix[q]])
    scales = spatial_scales([dict(x=torch.tensor(x[tt == t]), s=torch.tensor(s[tt == t])) for t in fit])
    info = dict(status='finished', task_scope='full_series_fitting', independent_forecast_test=False, protocol=f'{case}_counts_once_fullseries_HVG_PCA10', source=str(source), source_sha256=source_hash, source_layer='counts', source_X_used=False, source_shape=list(raw.shape), source_counts_integer_max_error=integer_error, normalization_identity_verified=True, fit_times=fit, excluded_times=[], n_train=len(ix), n_initial=int(sum(tt == fit[0])), n_virtual=0, n_hvg=len(hvg_genes), n_genes_source=len(genes), n_genes_after_cohort_filter=len(retained_genes), training_counts=training_stats, normalization=normalization, gene_standardization=False, pc_standardization=False, pca_n_samples=int(pca.n_samples_), pca_n_components=10, pca_random_state=0, pca_solver=pca.svd_solver, variance_ratio=pca.explained_variance_ratio_.tolist(), spatial_policy=spatial_policy, spatial_scales=scales, training_sha256=digest(out / 'training.npz'), transform_sha256=digest(out / 'transform.joblib'), analysis_files_sha256={p.name: digest(p) for p in sorted(analysis.glob('*.npz'))}, evaluation_note='Fitted reconstruction only; new coordinate basis, not comparable numerically to frozen forecast benchmark losses.', upstream_limit='Supplied source universe and original fixed QC retained; upstream selection not reversed.')
    atomic_json(out / 'ready.json', info)
    print('PREPROCESS_FINISHED', case, len(ix), scales, flush=True)
if __name__ == '__main__':
    main()
