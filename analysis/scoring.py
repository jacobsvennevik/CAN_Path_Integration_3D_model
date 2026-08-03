"""
All of the grid-field scoring code in one module, inheritance from G and Y in that file.

The disk way needs a run that was saved to a .npz file (record=True).

The RAM way skips the file. We run the network and add up the activity as we go,

TODO: the disk way and the and the RAM way is different as of now belive RAM is correct

"""

import json
import warnings
from dataclasses import dataclass
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist
from scipy.signal import correlate

from config import (
    AnalysisConfig, ExperimentConfig,
    world_to_normalized, world_to_flat_bins, world_to_flat_bins_3d,
)
from . import gongyu_scoring as gy


# Cube all scorers assume. Matches Gong & Yu [-1,1]^3.
LIM = ((-1, 1), (-1, 1), (-1, 1))

# Raw .npz keys written by the harness when record=True.
RAW_POS_KEY = "world_pos"
RAW_ACT_KEY = "S_tot_buffer"

BINS         = ExperimentConfig.ratemap_bins
SMOOTH_SIGMA = AnalysisConfig.smooth_sigma
AUTOCORR_TH  = AnalysisConfig.autocorr_th


@dataclass
class ScoringInput:
    """The (x, a) pair every scorer consumes. Invariants checked in `validate`."""
    x: np.ndarray                        # positions, each axis within [-1, 1]
    a: np.ndarray                        # activity, each cell in [0, 1]
    cell_idx: np.ndarray | None = None   # which raw columns survived subsampling
    meta: dict | None = None             # free-form provenance

    @property
    def T(self) -> int:
        return self.x.shape[0]

    @property
    def n(self) -> int:
        return self.a.shape[1]

def load_npz(npz_path: str) -> dict:
    """Read an .npz run file and return all arrays as a dict."""
    d = np.load(npz_path, mmap_mode="r")
    return {k: np.asarray(d[k]) for k in d.files}


def _normalise_activity(S: np.ndarray, mode: str = "divmax"):
    """Maps positive activity into [0, 1] per neuron.
    "divmax" preserves this is used in analysis following Skaggs spatial information.
    "minmax" stretches the full dynamic range
    """
    mn, mx = S.min(0), S.max(0)
    if mode == "minmax":
        a = (S - mn) / np.where(mx > mn, mx - mn, 1.0)
    elif mode == "divmax":
        a = S / np.where(mx > 0, mx, 1.0)
    else:
        raise ValueError(f"unknown norm mode {mode!r}; use 'divmax' or 'minmax'")
    a[np.isnan(a)] = 0.0
    np.clip(a, 0.0, 1.0, out=a)   # guard tiny float overshoot for validate()
    return a, mn, mx


def scoring_input_from_npz(npz_path: str, n_neuron: int = 20000,
                           seed: int = 0, norm: str = "divmax",
                           active_thresh: float = 1e-3,
                           env_size: float | None = None) -> "ScoringInput":
    """Runs the score, turns a saved .npz file into something the scoring pipeline can run on. This is done by:
    1. Loading
    2. Rescaling positions
    3. Normalising activity pr neuron
    4. Filter only active neurons and then sumbsamples them
    """
    d         = load_npz(npz_path)
    world_pos = d[RAW_POS_KEY]     # (T, 3)
    S         = d[RAW_ACT_KEY]     # (T, N), non-negative

    # work out env_size if we were not given one
    if env_size is None:
        cfg_path = npz_path.replace(".npz", "_config.json")
        try:
            with open(cfg_path) as fh:
                env_size = json.load(fh)["params"]["experiment"]["env_size"]
        except (FileNotFoundError, KeyError, TypeError):
            warnings.warn(
                f"scoring_input_from_npz: env_size not found in {cfg_path}; ",
                UserWarning, stacklevel=2,
            )

    if env_size is not None:
        # rescale positions to [-1, 1]^3 using the real box, keeps geometry correct
        x = world_to_normalized(world_pos, env_size)
    else:
        # fallback: just use the smallest box that holds the path (old way, not great)
        lo, hi = world_pos.min(0), world_pos.max(0)
        center = (hi + lo) / 2.0
        half   = np.max((hi - lo) / 2.0)
        x      = (world_pos - center) / half

    # subsample from active neurons only
    mn = S.min(0)
    mx = S.max(0)
    rng       = np.random.default_rng(seed)
    active    = (mx - mn) > active_thresh * mx.max()
    active_idx = np.where(active)[0]
    n_take    = min(n_neuron, len(active_idx))
    idx       = np.sort(rng.choice(active_idx, size=n_take, replace=False))

    # normalise activity to [0, 1] per neuron
    S_sub    = np.asarray(S[:, idx])
    a, _, _  = _normalise_activity(S_sub, mode=norm)

    return ScoringInput(
        x=x, a=a, cell_idx=idx,
        meta=dict(npz_path=npz_path, n_neuron=int(n_take), seed=int(seed),
                  norm=norm, n_active=int(active.sum()), n_total=int(S.shape[1]),
                  env_size=env_size),
    )



def activity_rate_map(si: ScoringInput, dims=(0, 1),
                      bins: int = BINS,
                      sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Mean activity per spatial bin.

    Works for both 2D (dims=(0,1)) and 3D (dims=(0,1,2)).

    Returns
    -------
    f : ndarray, shape (bins,)*len(dims) + (n,)
        Smoothed mean activity per bin per neuron.
    """
    ndim = len(dims)
    if ndim not in (2, 3):
        raise ValueError(f"dims must be length 2 or 3, got {ndim}")

    x   = si.x[:, list(dims)]
    lim = tuple(LIM[d] for d in dims)          # same range convention as occupancy()

    # visits per bin (unweighted) and summed activity per bin (weighted), per neuron
    counts, _ = np.histogramdd(x, bins=bins, range=lim)
    sums = np.stack(
        [np.histogramdd(x, bins=bins, range=lim, weights=si.a[:, c])[0]
         for c in range(si.n)],
        axis=-1,
    )

    # mean activity = summed activity / visits; empty bins -> 0
    f = sums / np.where(counts[..., None] > 0, counts[..., None], 1.0)

    # Smooth each neuron's map independently
    for c in range(si.n):
        f[..., c] = gaussian_filter(f[..., c], sigma=sigma)

    return f


def occupancy(si: ScoringInput, dims=(0, 1, 2),
              bins: int = BINS) -> np.ndarray:
    """Fraction of timesteps spent in each bin.

    dims controls whether a 2D or 3D occupancy map is returned.
    Must match the dims used in activity_rate_map.
    """
    lim = tuple(LIM[d] for d in dims)
    p, _ = np.histogramdd(si.x[:, list(dims)], bins=bins, range=lim,
                          density=False)
    return p / p.sum()


def autocorr2d(f: np.ndarray, th: float = AUTOCORR_TH) -> np.ndarray:
    """Standardised 2-D autocorrelation.

    Parameters
    ----------
    f  : (bins, bins, n)  smoothed rate map
    th : values below this threshold are zeroed in the output

    Returns
    -------
    ac : (2*bins-1, 2*bins-1, n), non-negative
    """
    if f.ndim == 2:
        f = f[..., None]
    n   = f.shape[0] * f.shape[1]
    std = f.std(axis=(0, 1))
    std = np.where(std < 1e-10, 1.0, std)
    f_  = (f - f.mean(axis=(0, 1))) / std
    out = []
    for i in range(f.shape[-1]):
        ac = correlate(f_[..., i], f_[..., i], mode="full") / n
        ac[ac < th] = 0.0          # suppress weak correlations (display threshold)
        np.clip(ac, 0.0, None, out=ac)   # guarantee non-negative for gridness assert
        out.append(ac)
    return np.stack(out, axis=-1)


def hex_gridness_2d(ac2d: np.ndarray, peak_thresh: float = 0.1) -> tuple:
    """(hgs, sgs) hexagonal/square gridness for one 2-D autocorrelogram.

    Finds the inner-ring radius directly from the 2-D autocorrelation peaks rather
    than from the radial-mean profile: the radial *mean* averages a real 6-peak
    ring together with its 6 troughs into a smooth monotonic decay, so
    `gy.peak` misses the ring and the score collapses to NaN even when the ring is
    plainly visible. Locating the peaks in 2-D avoids that failure mode.

    Returns (nan, nan) when no ring is found (genuinely no hexagonal structure).
    """
    from scipy.ndimage import maximum_filter
    d = ac2d.shape[0]
    c = d // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1] if d % 2 else np.mgrid[-c:c, -c:c]
    r = np.sqrt(xx ** 2 + yy ** 2)
    is_peak = (ac2d == maximum_filter(ac2d, size=3)) & (ac2d > peak_thresh) & (r > 2)
    radii = r[is_peak]
    if radii.size < 3:
        return np.nan, np.nan                       # no ring -> not a grid
    ring_r = np.median(np.sort(radii)[:6])          # six nearest peaks = inner ring
    lb, ub = int(round(0.5 * ring_r)), min(int(round(1.5 * ring_r)), c)
    if ub <= lb:
        return np.nan, np.nan
    return gy.gridness(ac2d, lb, ub)

def autocorrelation_1cell(f_cell: np.ndarray, th: float = AUTOCORR_TH) -> np.ndarray:
    """Standardised 3-D autocorrelation of a single cell's rate map.

    Parameters
    ----------
    f_cell : (b, b, b) one neuron's smoothed rate map

    Returns
    -------
    ac : (2b-1, 2b-1, 2b-1) non-negative autocorrelogram for that cell.
    """
    from scipy.signal import correlate
    nbins = f_cell.size
    std = f_cell.std()
    std = std if std > 1e-10 else 1.0
    f_ = (f_cell - f_cell.mean()) / std
    ac = correlate(f_, f_, mode="full") / nbins
    ac[ac < th] = 0.0
    return ac


def chi_1cell(ac_cell: np.ndarray, align: bool = False) -> np.ndarray:
    """(3,) chi scores (fcc, hcp, col) for one cell's 3-D autocorrelogram.

    Parameters
    ----------
    ac_cell : (d, d, d) non-negative autocorrelation cube for one cell.
    align   : if True, find the best hexagonal plane (gy.best_plane) and rotate the
              cube before scoring.

    """
    cube = ac_cell[..., None]                  # (d, d, d, 1) as chi_score expects
    if align:
        try:
            az, al, _ = gy.best_plane(cube)
            cube = _rotate_cube_to_horizontal(cube, az[0], al[0])
            cube[cube < 0] = 0.0               # rotation interpolation can dip <0
        except (AssertionError, ValueError):
            return np.full(3, np.nan)
    try:
        fcc, hcp, col = gy.chi_score(cube)
        return np.array([fcc[0], hcp[0], col[0]])
    except AssertionError:
        return np.full(3, np.nan)              # peak() found no clean ring


def _rotate_cube_to_horizontal(cube: np.ndarray, az: float, al: float) -> np.ndarray:
    """Resample an autocorrelation cube so the (az, al) plane becomes horizontal.

    Uses the same rotation convention as gy.rot_x / gy.oblique_slice, so the
    aligned cube is consistent with chi_score's fixed plane indices.
    """
    from scipy.interpolate import RegularGridInterpolator
    d = cube.shape[0]
    grid = np.linspace(-1, 1, d)
    interp = RegularGridInterpolator((grid, grid, grid), cube[..., 0],
                                     bounds_error=False, fill_value=0.0)
    gx, gy_, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([gx.ravel(), gy_.ravel(), gz.ravel()], axis=-1)
    # inverse rotation: sample the original cube at rotated coordinates
    pts_rot = gy.rot_x(pts, azimuth=-az, altitude=-al)
    out = interp(pts_rot).reshape(d, d, d)
    return out[..., None]


def mra_1cell(ac_cell: np.ndarray) -> np.ndarray:
    """(rmax,) modified radial autocorrelation (sum / sqrt(N(r))) for one cell."""
    rmax = ac_cell.shape[0] // 2
    return gy.autocorr_radial3d(ac_cell, rmax, method="mean_comp")


def global_order_1cell(ac_cell: np.ndarray, az_precision: int = 48,
                       al_precision: int = 24, radial_method: str = "max") -> float:
    """Orientation-invariant global-order score for one cell.
    """
    hgs_map, _ = gy.gridness_map(ac_cell[..., None], az_precision=az_precision,
                                 al_precision=al_precision, al_max=np.pi / 2,
                                 radial_method=radial_method, hex_only=True)
    return float(hgs_map.max())


def _spatial_info(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Skaggs spatial information (bits/sample), dimension-agnostic.

    """
    ndim = f.ndim - 1
    ax = tuple(range(ndim))
    p = p[..., None]
    lam_bar = (f * p).sum(axis=ax)                 # mean rate per cell
    f_ = f / np.where(lam_bar != 0, lam_bar, 1.0)  # lambda_i / lambda_bar
    f_ = np.where(f_ == 0, 1.0, f_)                # lim x->0 x log x = 0
    return (np.log2(f_) * p * f_).sum(axis=ax)


def _sparsity_idx(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Sparsity index (E[f]^2 / E[f^2]), does not care about dimension"""
    ndim = f.ndim - 1
    ax = tuple(range(ndim))
    p = p[..., None]
    e_f = (f * p).sum(axis=ax)
    e_f2 = (f ** 2 * p).sum(axis=ax)
    return e_f ** 2 / np.where(e_f2 != 0, e_f2, 1.0)


def info_scores(si: ScoringInput, p: np.ndarray, f: np.ndarray,
                n_shuffle: int = 50, seed: int = 0,
                bins: int = BINS, sigma: float = SMOOTH_SIGMA) -> dict:
    """Spatial information, sparsity, and their shuffle Z-scores.
 
    Null: permute each neuron's activity in time (breaks the position-activity
    link) and recompute the rate map. Matches G&Y's si_shuffle. 2-D or 3-D.
 
    p : occupancy map, must match f's spatial dims.
    f : rate map, (bins,)*ndim + (n,).
    """
    rng   = np.random.default_rng(seed)
    ndim  = f.ndim - 1             # spatial dimensions of the rate map
    dims  = tuple(range(ndim))

    sinfo = _spatial_info(p, f)
    sidx  = _sparsity_idx(p, f)

    sinfo_sf = np.zeros((n_shuffle, si.n))
    sidx_sf  = np.zeros((n_shuffle, si.n))

    for i in range(n_shuffle):
        # Permute each neuron's activity independently (temporal shuffle)
        a_sf = np.stack(
            [rng.permutation(si.a[:, c]) for c in range(si.n)], axis=1
        )
        si_sf = ScoringInput(x=si.x, a=a_sf, cell_idx=si.cell_idx)
        f_sf  = activity_rate_map(si_sf, dims=dims, bins=bins, sigma=sigma)
        sinfo_sf[i] = _spatial_info(p, f_sf)
        sidx_sf[i]  = _sparsity_idx(p, f_sf)

    sinfo_z = (sinfo - sinfo_sf.mean(0)) / np.where(
        sinfo_sf.std(0) > 1e-10, sinfo_sf.std(0), 1.0)
    sidx_z  = (sidx  - sidx_sf.mean(0))  / np.where(
        sidx_sf.std(0)  > 1e-10, sidx_sf.std(0),  1.0)

    return dict(sinfo=sinfo, sidx=sidx, sinfo_z=sinfo_z, sidx_z=sidx_z)


def inter_field_distances(si: ScoringInput, activity_pct: float = 90.0,
                          bandwidth: float = 0.25, min_bin_freq: int = 25,
                          min_cluster_size: int = 30,
                          ignore_range: float = 0.95) -> dict:
    """Find the distance between two firing fields.
    """
    out = {}
    for k in range(si.n):
        threshold  = np.percentile(si.a[:, k], activity_pct)
        active_pos = si.x[si.a[:, k] >= threshold]
        if len(active_pos) < min_bin_freq:        # too few points to seed a bin
            out[k] = np.array([])
            continue
        try:
            _, _, centers = gy.ms_cluster(
                active_pos, bandwidth=bandwidth, min_bin_freq=min_bin_freq,
                min_cluster_size=min_cluster_size, ignore_range=ignore_range,
                plot=False)
        except ValueError:
            out[k] = np.array([])                 # MeanShift found no seeds
            continue
        out[k] = pdist(centers) if len(centers) >= 2 else np.array([])
    return out


def _rate_map_from_accumulator(sums: np.ndarray, counts: np.ndarray,
                               sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Smoothed mean-activity rate map from online accumulators.

    Works for any spatial dimensionality:
        2-D input : sums (b, b, n),   counts (b, b)
        3-D input : sums (b, b, b, n), counts (b, b, b)

    Returns f of the same shape as sums.
    """
    denom = np.where(counts > 0, counts, 1.0)[..., None]
    f = sums / denom
    for c in range(f.shape[-1]):
        f[..., c] = gaussian_filter(f[..., c].astype(np.float32), sigma=sigma)
    return f

def _score_one_cell(f_k, align, global_order, go_precision):
    """Score one neuron's rate map: chi, MRA, and (if asked) global order."""
    ac_k  = autocorrelation_1cell(f_k)
    chi_k = chi_1cell(ac_k, align=align)
    mra_k = mra_1cell(ac_k)
    go_k  = np.nan
    if global_order:
        go_k = global_order_1cell(ac_k, az_precision=go_precision[0],
                                  al_precision=go_precision[1])
    return chi_k, mra_k, go_k


def _stream_structure_scores(
    f: np.ndarray,
    bins: int,
    align: bool = False,
    global_order: bool = False,
    go_precision: tuple = (48, 24),
    n_jobs: int = 1,
) -> tuple:
    """Work out the chi / MRA / (if asked) global-order scores, one neuron at a time.

    Both score_run (the disk way) and score_3d_from_map (the RAM way) lean on this.
    We only ever keep one neuron's big autocorrelation cube in memory at once, so
    this stays light even with lots of neurons.

    Returns
    -------
    chi_out    (3, n)   the fcc / hcp / col structure scores
    mra_out    (n, rmax)
    go_out     (n,) or None
    ring_found (n,) bool
    no_ring    (n,) bool
    n_failed   int
    """
    n    = f.shape[-1]
    rmax = (2 * bins - 1) // 2
    chi_out = np.full((3, n), np.nan)
    mra_out = np.zeros((n, rmax))
    go_out  = np.full(n, np.nan) if global_order else None

    if n_jobs == 1:
        # one neuron at a time, the original way
        for k in range(n):
            chi_k, mra_k, go_k = _score_one_cell(f[..., k], align,
                                                 global_order, go_precision)
            chi_out[:, k] = chi_k
            mra_out[k]    = mra_k
            if global_order:
                go_out[k] = go_k
    else:
        # spread neurons over cores; pin each worker to one thread so
        # scipy/BLAS don't fight over cores
        from joblib import Parallel, delayed, parallel_backend
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_score_one_cell)(f[..., k].copy(), align,
                                         global_order, go_precision)
                for k in range(n)
            )
        for k, (chi_k, mra_k, go_k) in enumerate(results):
            chi_out[:, k] = chi_k
            mra_out[k]    = mra_k
            if global_order:
                go_out[k] = go_k

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # quiet the all-NaN slice moan
        chi_max = np.nanmax(chi_out, axis=0)

    # tell apart "we looked and there was no ring" (chi pinned to 0) from a real
    # measured-low score, and from an outright fail (NaN). chi sitting at 0 is NOT the
    # same evidence as a low number we actually measured.
    ring_found = np.isfinite(chi_max) & (np.abs(chi_max) > 1e-9)
    no_ring    = np.isfinite(chi_max) & (np.abs(chi_max) <= 1e-9)
    return chi_out, mra_out, go_out, ring_found, no_ring, int(np.isnan(chi_max).sum())


def _online_shuffle_zscores(
    shuf_sums: np.ndarray,
    scored_idx: np.ndarray,
    denom: np.ndarray,
    p: np.ndarray,
    sigma: float,
    sinfo: np.ndarray,
    sidx: np.ndarray,
) -> tuple:
    """Work out sinfo_z / sidx_z using the slide-in-time shuffle (RAM way only).

    This is a different test from the one in info_scores (which shuffles fully at
    random). Sliding each neuron forward in time keeps the neuron's own slow wiggle,
    which makes it a tougher test for CAN activity. So do not put these Z-scores
    head to head with the full-shuffle ones from the disk way.

    Parameters
    ----------
    shuf_sums  : (n_shuffle, *spatial, n_sub)  the shuffle maps we built earlier
    scored_idx : which neuron columns to score
    denom      : safe visit-count divider, shape (*spatial, 1)
    p          : fraction of time spent in each bin, shape (*spatial,)
    sigma      : how much to smooth, in bins
    sinfo, sidx: the real (un-shuffled) values, each (n_scored,)

    Returns
    -------
    (sinfo_z, sidx_z)  both (n_scored,) float64
        Rule of thumb: Z of 2.58 or more means 99% sure (Gong and Yu, Figure 4).
    """
    n_shuffle = shuf_sums.shape[0]
    n_scored  = len(scored_idx)
    sinfo_shuf = np.zeros((n_shuffle, n_scored), dtype=np.float32)
    sidx_shuf  = np.zeros((n_shuffle, n_scored), dtype=np.float32)

    for j in range(n_shuffle):
        f_j = shuf_sums[j][..., scored_idx] / denom
        for c in range(n_scored):
            f_j[..., c] = gaussian_filter(f_j[..., c].astype(np.float32), sigma=sigma)
        sinfo_shuf[j] = _spatial_info(p, f_j).astype(np.float32)
        sidx_shuf[j]  = _sparsity_idx(p, f_j).astype(np.float32)
        del f_j

    sinfo_z = (sinfo - sinfo_shuf.mean(0)) / (sinfo_shuf.std(0) + 1e-9)
    sidx_z  = (sidx  - sidx_shuf.mean(0))  / (sidx_shuf.std(0)  + 1e-9)
    return sinfo_z.astype(np.float64), sidx_z.astype(np.float64)


def score_2d(npz_path: str, n_neuron: int = 5000, seed: int = 0,
             bins: int = BINS, sigma: float = SMOOTH_SIGMA,
             norm: str = "divmax") -> dict:
    """End-to-end 2-D baseline: project to xy, score hexagonality.

    Sucess: autocorrelation shows a 6-fold ring for most of the included neurons.
    
    NOT IN USE ANYMORE
    """
    si = scoring_input_from_npz(npz_path, n_neuron=n_neuron, seed=seed, norm=norm)
    assert si.x[:, 2].std() < 1e-3, \
        f"score_2d expects a flat-floor run; z std = {si.x[:, 2].std():.3g}"

    f  = activity_rate_map(si, dims=(0, 1), bins=bins, sigma=sigma)  # (b,b,n)
    ac = autocorr2d(f)                                                # (2b-1,2b-1,n)
    p  = occupancy(si, dims=(0, 1), bins=bins)                        # (b,b)

    # per-cell hexagonal gridness: the actual baseline pass/fail number, not just
    # an autocorrelogram to eyeball. SGS should sit low for a real hexagonal grid.
    hgs = np.full(si.n, np.nan)
    sgs = np.full(si.n, np.nan)
    for k in range(si.n):
        hgs[k], sgs[k] = hex_gridness_2d(ac[..., k])

    return dict(
        si        = si,
        f         = f,
        ac        = ac,
        p         = p,
        hgs       = hgs,                    # hexagonal gridness per cell (grid if > ~0.3)
        sgs       = sgs,                    # square gridness (should be low for hex)
        ring_frac = float(np.isfinite(hgs).mean()),   # fraction with a detectable ring
        peak      = f.max(axis=(0, 1)),     # peak activity per neuron
        occupancy = float((p > 0).mean()),
    )


def score_run(npz_path: str, n_neuron: int = 300, seed: int = 0,
              bins: int = BINS, sigma: float = SMOOTH_SIGMA,
              align: bool = False, norm: str = "divmax",
              global_order: bool = False, go_precision: tuple = (48, 24),
              occ_warn: float = 0.5,
              env_size: float | None = None, n_jobs: int = 1) -> dict:
    """Full 3-D scoring pipeline (the disk way, memory-safe, streams one cell at a time).

    Returns chi structure scores, MRA, spatial information, sparsity, and
    inter-field distances per neuron.

    env_size : how big the real box is, in metres; just passed along to
               scoring_input_from_npz. Leave as None to read it from the
               _config.json file next to the run (the recommended way).
    """
    si  = scoring_input_from_npz(npz_path, n_neuron=n_neuron, seed=seed,
                                 norm=norm, env_size=env_size)
    f   = activity_rate_map(si, dims=(0, 1, 2), bins=bins, sigma=sigma)
    p   = occupancy(si, dims=(0, 1, 2), bins=bins)
    occ = float((p > 0).mean())

    if occ < occ_warn:
        warnings.warn(
            f"occupancy {occ:.1%}: map is sparse — chi/MRA reflect coverage "
            "more than tuning. Lengthen the trajectory."
        )
    if align:
        warnings.warn(
            "align=True is miscalibrated against chi_score's fixed template and "
            "can collapse a cleanly-ordered cell to ~0 (see chi_1cell). Prefer "
            "align=False + global_order=True for an orientation-invariant readout."
        )
    if n_neuron > 5000:
        warnings.warn(f"n_neuron={n_neuron} > 5000: scoring will be slow.")

    chi_out, mra_out, go_out, ring_found, no_ring, n_failed = \
        _stream_structure_scores(f, bins, align=align, global_order=global_order, 
                                 go_precision=go_precision, n_jobs=n_jobs)

    info = info_scores(si, p, f, bins=bins, sigma=sigma, seed=seed)
    ifd  = inter_field_distances(si)

    out = dict(
        idx        = si.cell_idx,
        chi        = chi_out,
        mra        = mra_out,
        ring_found = ring_found,
        no_ring    = no_ring,
        n_failed   = n_failed,
        **info,
        ifd        = ifd,
        peak       = f.max(axis=(0, 1, 2)),
        occupancy  = occ,
        aligned    = align,
    )
    if global_order:
        out["go"] = go_out
    return out

def run_with_online_ratemap(exp, g, bins: int = 40, n_warmup: int = 100,
                            mode: str = "integrate",
                            trajectory=None,
                            n_shuffle: int = 0,
                            shuffle_min_lag_frac: float = 0.1,
                            seed: int = 0) -> dict:
    """Memory light 2-D run (just the XY floor).

    We add the activity up into a (bins², N) map on the device as we go, so the huge
    per-step buffer is never needed. One trip back to the CPU at the very end. If you
    want the full 3-D version use run_3d_online instead.

    Parameters
    ----------
    exp        : a BaseExperiment subclass
    g          : gravity vector
    trajectory : optional (world_pos, v_body_seq, torus_gt) tuple.
                 Hand this in to reuse a path you already have from
                 run_experiment(), so we do not run the network a second time
                 just to get the same path back.
    n_shuffle  : number of time-lag shuffle maps to build online (0 = disabled).
                 When > 0, shuf_sums is returned and can be passed to
                 score_2d_from_map for sinfo_z / sidx_z without storing the
                 full per-step activity buffer.
    shuffle_min_lag_frac : minimum lag as fraction of T (mirrors run_3d_online).
    seed       : RNG seed for lag sampling.
    """
    from model.path_integration import PathIntegrator

    cfg   = exp.config.experiment
    integ = PathIntegrator(qan=exp.qan, **exp.integrator_kwargs)
    bk    = integ.backend
    dev   = getattr(bk, "device", torch.device("cpu"))

    if trajectory is not None:
        world_pos, v_body_seq, torus_gt = trajectory
    else:
        world_pos, v_body_seq, torus_gt = exp.generate_trajectory()

    T, N = world_pos.shape[0], bk.S.shape[1]
    integ.reset(torus_gt[0])
    if mode == "integrate" and n_warmup:
        integ.warmup(n_warmup)

    # arena-anchored 2-D bin number for each step (XY only)
    flat_2d = world_to_flat_bins(world_pos, cfg.env_size, bins)     # (T,) int64

    sums   = torch.zeros((bins * bins, N), dtype=torch.float32, device=dev)
    counts = torch.zeros(bins * bins,      dtype=torch.float32, device=dev)

    # shuffle accumulators — only allocated when asked for
    shuf_sums_d = None
    rng = np.random.default_rng(seed)
    if n_shuffle > 0:
        min_lag = max(1, int(T * shuffle_min_lag_frac))
        if min_lag >= T:
            warnings.warn(
                f"run_with_online_ratemap: T={T} too short for n_shuffle={n_shuffle} "
                f"(min_lag={min_lag} >= T). Disabling shuffle.",
                UserWarning, stacklevel=2,
            )
            n_shuffle = 0
        else:
            lags        = rng.integers(min_lag, T, size=n_shuffle)
            shuf_sums_d = torch.zeros(
                (n_shuffle, bins * bins, N), dtype=torch.float32, device=dev)

    theta_hist = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        if mode == "integrate":
            theta_hist[t] = integ.step(v_body_seq[t], g)
        else:
            bk.reset(torus_gt[t])
            theta_hist[t] = torus_gt[t]
        s = bk.S.mean(dim=0).squeeze()          # (N,) on device, no per-step CPU copy
        b = int(flat_2d[t])
        sums[b]   += s
        counts[b] += 1.0
        if shuf_sums_d is not None:
            for j in range(n_shuffle):
                b_shuf = int(flat_2d[(t + lags[j]) % T])
                shuf_sums_d[j, b_shuf] += s

    shuf_sums = None
    if shuf_sums_d is not None:
        shuf_sums = shuf_sums_d.cpu().numpy().reshape(
            n_shuffle, bins, bins, N)

    return dict(
        sums=sums.cpu().numpy().reshape(bins, bins, N),
        counts=counts.cpu().numpy().reshape(bins, bins),
        shuf_sums=shuf_sums,
        bins=bins, world_pos=world_pos, torus_gt=torus_gt, theta_hist=theta_hist,
    )

def run_3d_online(
    exp,
    g_vec: np.ndarray,
    *,
    trajectory=None,
    bins: int = 25,
    n_sub: int = 300,
    seed: int = 0,
    n_warmup: int = 100,
    active_thresh: float = 1e-3,
    n_shuffle: int = 0,
    shuffle_min_lag_frac: float = 0.1,
) -> dict:
    """RAM-only 3-D run. Never makes the big S_tot_buffer.

    We step the network one timestep at a time and add the activity into a
    (bins³, n_sub) tensor on the device. One trip back to the CPU at the end. If you
    ask for shuffle maps too, score_3d_from_map can then work out sinfo_z / sidx_z
    without ever needing the full per-step activity buffer.

    """
    from model.path_integration import PathIntegrator

    g_vec = np.asarray(g_vec, dtype=float)
    cfg   = exp.config.experiment

    # get the path: reuse one if handed in, otherwise make a fresh one
    if trajectory is not None:
        world_pos, v_body_seq, torus_gt = trajectory
    else:
        world_pos, v_body_seq, torus_gt = exp.generate_trajectory()
    T = world_pos.shape[0]

    # the thing that steps the network
    integ = PathIntegrator(qan=exp.qan, **exp.integrator_kwargs)
    bk    = integ.backend
    dev   = getattr(bk, "device", torch.device("cpu"))

    # pick which neurons to follow up front, keeps the accumulator small
    N        = bk.S.shape[1]
    rng      = np.random.default_rng(seed)
    sub_idx  = np.sort(rng.choice(N, size=min(n_sub, N), replace=False))
    sub_t    = torch.tensor(sub_idx, dtype=torch.long, device=dev)
    n_sub_actual = len(sub_idx)

    # the running totals, kept on the device
    sums_d   = torch.zeros((bins ** 3, n_sub_actual), dtype=torch.float32, device=dev)
    counts_d = torch.zeros(bins ** 3,                 dtype=torch.float32, device=dev)

    # extra totals for the shuffle, only if asked for
    shuf_sums_d = None
    lags        = None
    if n_shuffle > 0:
        min_lag = max(1, int(T * shuffle_min_lag_frac))
        if min_lag >= T:
            warnings.warn(
                f"run_3d_online: T={T} too short for n_shuffle={n_shuffle} "
                f"(min_lag={min_lag} ≥ T). Disabling shuffle.",
                UserWarning, stacklevel=2,
            )
            n_shuffle = 0
        else:
            lags        = rng.integers(min_lag, T, size=n_shuffle)
            shuf_sums_d = torch.zeros(
                (n_shuffle, bins ** 3, n_sub_actual),
                dtype=torch.float32, device=dev,
            )

    # work out which 3-D box each step lands in, all at once
    flat3d = world_to_flat_bins_3d(world_pos, cfg.env_size, bins)   # (T,) int64

    # put the bump at the start and let things settle
    integ.reset(torus_gt[0])
    integ.warmup(n_warmup)

    # the main loop: step, then drop the activity into its box
    theta_hist = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        theta_hist[t] = integ.step(v_body_seq[t], g_vec)

        s = bk.S.mean(dim=0).squeeze()     # (N,) on device
        b = int(flat3d[t])
        sums_d[b]   += s[sub_t]
        counts_d[b] += 1.0

        if shuf_sums_d is not None:
            for j in range(n_shuffle):
                b_shuf = int(flat3d[(t + lags[j]) % T])
                shuf_sums_d[j, b_shuf] += s[sub_t]

    # one trip back to the CPU
    sums   = sums_d.cpu().numpy().reshape(bins, bins, bins, n_sub_actual)
    counts = counts_d.cpu().numpy().reshape(bins, bins, bins)

    shuf_sums = None
    if shuf_sums_d is not None:
        shuf_sums = shuf_sums_d.cpu().numpy().reshape(
            n_shuffle, bins, bins, bins, n_sub_actual
        )

    # mark the neurons that actually did something, and grab the filter's history
    span        = sums.reshape(-1, n_sub_actual).max(0) - sums.reshape(-1, n_sub_actual).min(0)
    active_mask = span > active_thresh * (span.max() + 1e-12)

    n_hat_hist = np.array(integ.history["n_hat"])
    gap_hist   = np.array(integ.history["z2"]) - np.array(integ.history["z1"])

    return dict(
        sums=sums, counts=counts,
        sub_idx=sub_idx, active_mask=active_mask,
        world_pos=world_pos, torus_gt=torus_gt,
        theta_hist=theta_hist, bins=bins,
        n_hat_hist=n_hat_hist, gap_hist=gap_hist,
        shuf_sums=shuf_sums, n_shuffle=n_shuffle,
    )


def score_3d_from_map(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    shuf_sums: np.ndarray | None = None,
    sigma: float = SMOOTH_SIGMA,
    align: bool = False,
    global_order: bool = False,
    go_precision: tuple = (48, 24),
    occ_warn: float = 0.5,
    occ_error: float = 0.15,
    active_mask: np.ndarray | None = None,
    n_jobs: int = 1, 
) -> dict:
    """Score a 3-D rate map we already built. No .npz file, no ScoringInput needed.

    Note (Gong and Yu, Figure 4): Z of 2.58 or more.
    """
    bins    = sums.shape[0]
    n_total = sums.shape[-1]

    # which neurons are we scoring
    scored_idx = np.where(active_mask)[0] if active_mask is not None \
                 else np.arange(n_total)

    if len(scored_idx) == 0:
        warnings.warn("score_3d_from_map: no active neurons to score.", UserWarning, stacklevel=2)
        return dict(chi=None, mra=None, ring_found=None, no_ring=None, n_failed=0,
                    sinfo=None, sidx=None, sinfo_z=None, sidx_z=None, ifd=None,
                    peak=None, occupancy=0.0, reliable=False,
                    scored_idx=scored_idx, bins=bins, aligned=align)

    # turn the totals into a smoothed rate map
    sums_sub = sums[..., scored_idx].copy()
    denom    = np.where(counts > 0, counts, 1.0)[..., None]
    f        = _rate_map_from_accumulator(sums_sub, counts, sigma=sigma)

    # how much of the box did we actually cover, and can we trust it
    p   = counts / (counts.sum() + 1e-30)
    occ = float((p > 0).mean())

    reliable = True
    if occ < occ_error:
        warnings.warn(
            f"score_3d_from_map: occupancy {occ:.1%} < {occ_error:.0%}. "
            f"Structural scores are NaN-filled (reliable=False). "
            f"Minimum T for occ ≥ {occ_error:.0%} at bins={bins}: "
            f"≳ {int(occ_error * bins**3)} steps.  "
            f"Try bins=12 for short exploratory runs.",
            UserWarning, stacklevel=2,
        )
        reliable = False
        n_scored = len(scored_idx)
        rmax     = (2 * bins - 1) // 2
        sinfo    = _spatial_info(p, f)
        sidx     = _sparsity_idx(p, f)
        sinfo_z, sidx_z = None, None
        if shuf_sums is not None:
            sinfo_z, sidx_z = _online_shuffle_zscores(
                shuf_sums, scored_idx, denom, p, sigma, sinfo, sidx)
        out = dict(
            chi=np.full((3, n_scored), np.nan),
            mra=np.full((n_scored, rmax), np.nan),
            ring_found=np.zeros(n_scored, bool), no_ring=np.zeros(n_scored, bool),
            n_failed=n_scored,
            sinfo=sinfo, sidx=sidx, sinfo_z=sinfo_z, sidx_z=sidx_z, ifd=None,
            peak=f.max(axis=(0, 1, 2)), occupancy=occ, reliable=False,
            scored_idx=scored_idx, bins=bins, aligned=align,
        )
        return out

    elif occ < occ_warn:
        warnings.warn(
            f"score_3d_from_map: occupancy {occ:.1%} — map is sparse. "
            f"chi/MRA may reflect coverage more than tuning. "
            f"T ≳ {int(2 * bins**3)} steps needed for occ > {occ_warn:.0%}.",
            UserWarning, stacklevel=2,
        )

    # the spatial info numbers
    sinfo = _spatial_info(p, f)
    sidx  = _sparsity_idx(p, f)

    sinfo_z, sidx_z = None, None
    if shuf_sums is not None:
        sinfo_z, sidx_z = _online_shuffle_zscores(
            shuf_sums, scored_idx, denom, p, sigma, sinfo, sidx)

    # the structure scores, one neuron at a time
    chi_out, mra_out, go_out, ring_found, no_ring, n_failed = \
        _stream_structure_scores(f, bins, align=align,
                                 global_order=global_order, go_precision=go_precision, n_jobs=n_jobs)

    out = dict(
        chi=chi_out, mra=mra_out,
        ring_found=ring_found, no_ring=no_ring, n_failed=n_failed,
        sinfo=sinfo, sidx=sidx,
        sinfo_z=sinfo_z, sidx_z=sidx_z,     # explicit None when no shuf_sums
        ifd=None,                             # unavailable on online path
        peak=f.max(axis=(0, 1, 2)),
        occupancy=occ, reliable=reliable,
        scored_idx=scored_idx, bins=bins, aligned=align,
    )
    if global_order:
        out["go"] = go_out
    return out

def run_and_score_3d(
    exp,
    g_vec: np.ndarray,
    *,
    bins: int = 25,
    n_sub: int = 300,
    seed: int = 0,
    n_warmup: int = 100,
    sigma: float = SMOOTH_SIGMA,
    global_order: bool = False,
    n_shuffle: int = 0,
    n_jobs: int = 1, 
) -> dict:
    """
    One-shot: run_3d_online then score_3d_from_map.

    Returns a merged dict with all keys from both functions, plus:
        grid_like  (n_scored,) bool
        grid_like_method  str  — which metric built grid_like

    grid_like is built from global_order scores ('go') when global_order=True,
    which is the orientation-invariant readout matching Gong & Yu §2.6.3.
    When global_order=False, grid_like falls back to the chi-based triage verdict,
    which is FRAME-DEPENDENT (calibrated to the 8° prototype orientation).
    For any result you intend to publish, use global_order=True and validate
    against prototypes.reference_scores() at your run's bins.
    """
    raw    = run_3d_online(
        exp, g_vec, bins=bins, n_sub=n_sub, seed=seed,
        n_warmup=n_warmup, n_shuffle=n_shuffle,
    )
    scores = score_3d_from_map(
        raw["sums"], raw["counts"],
        shuf_sums=raw["shuf_sums"],
        sigma=sigma, global_order=global_order,
        active_mask=raw["active_mask"],
        n_jobs=n_jobs,
    )

    if not scores or scores.get("chi") is None:
        return {**raw, **scores, "grid_like": np.zeros(0, bool),
                "grid_like_method": "none (no scored cells)"}

    # Build grid_like from the most reliable available metric
    if global_order and scores.get("go") is not None:
        go = scores["go"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            grid_like = np.isfinite(go) & (go > 0)
        grid_like_method = "global_order (orientation-invariant)"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chi_max = np.nanmax(scores["chi"], axis=0)
        grid_like = scores.get("ring_found", np.zeros(0, bool)) & (chi_max > 0)
        grid_like_method = (
            "chi_triage (FRAME-DEPENDENT: calibrated to 8° prototype orientation). "
            "Call with global_order=True for the orientation-invariant verdict."
        )
        n_pos = grid_like.sum()
        print(
            f"[run_and_score_3d] grid_like ({n_pos} / {len(grid_like)} cells) is "
            f"chi-based triage.  Pass global_order=True to score those {n_pos} "
            f"positives with the orientation-invariant readout."
        )

    return {**raw, **scores, "grid_like": grid_like,
            "grid_like_method": grid_like_method}
    
def score_2d_from_map(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    sigma: float = SMOOTH_SIGMA,
    active_mask: np.ndarray | None = None,
    shuf_sums: np.ndarray | None = None,
    occ_warn: float = 0.3,
) -> dict:
    """Score a 2-D rate map already built by run_with_online_ratemap.

    Parameters
    ----------
    shuf_sums : (n_shuffle, bins, bins, n_sub) or None
        Pre-built time-lag shuffle maps from the online run (result.shuf_sums).
        When provided, sinfo_z and sidx_z are computed via _online_shuffle_zscores.
        When None, sinfo_z and sidx_z are returned as None.
    """
    bins    = sums.shape[0]
    n_total = sums.shape[-1]

    scored_idx = (np.where(active_mask)[0] if active_mask is not None
                  else np.arange(n_total))

    f     = _rate_map_from_accumulator(sums[..., scored_idx], counts, sigma=sigma)
    p     = counts / (counts.sum() + 1e-30)
    denom = np.where(counts > 0, counts, 1.0)[..., None]
    occ   = float((p > 0).mean())

    if occ < occ_warn:
        warnings.warn(
            f"score_2d_from_map: occupancy {occ:.1%} — map is sparse. "
            f"HGS/ring results may reflect coverage rather than tuning.",
            UserWarning, stacklevel=2,
        )

    ac       = autocorr2d(f)          # (2*bins-1, 2*bins-1, n_scored)
    n_scored = len(scored_idx)
    hgs      = np.full(n_scored, np.nan)
    sgs      = np.full(n_scored, np.nan)
    for k in range(n_scored):
        hgs[k], sgs[k] = hex_gridness_2d(ac[..., k])

    ring_found = np.isfinite(hgs)
    grid_like  = ring_found & (hgs > 0.3)

    sinfo = _spatial_info(p, f)
    sidx  = _sparsity_idx(p, f)

    sinfo_z, sidx_z = None, None
    if shuf_sums is not None:
        sinfo_z, sidx_z = _online_shuffle_zscores(
            shuf_sums, scored_idx, denom, p, sigma, sinfo, sidx)

    return dict(
        f          = f,
        ac         = ac,
        p          = p,
        hgs        = hgs,
        sgs        = sgs,
        grid_like  = grid_like,
        ring_frac  = float(ring_found.mean()),
        sinfo      = sinfo,
        sidx       = sidx,
        sinfo_z    = sinfo_z,
        sidx_z     = sidx_z,
        peak       = f.max(axis=(0, 1)),
        occupancy  = occ,
        n_active   = n_scored,
        scored_idx = scored_idx,
        bins       = bins,
    )
    
def run_and_score_2d(
    exp,
    g_vec: np.ndarray,
    *,
    bins: int = 40,
    n_warmup: int = 100,
    sigma: float = SMOOTH_SIGMA,
    trajectory=None,
) -> dict:
    """One-shot: run_with_online_ratemap then score_2d_from_map.

    The 2-D analogue of run_and_score_3d. Useful as a fast hexagonality
    sanity check on a flat-floor run without needing the full 3-D pipeline.
    """
    raw    = run_with_online_ratemap(exp, g_vec, bins=bins,
                                     n_warmup=n_warmup, trajectory=trajectory)
    scores = score_2d_from_map(raw["sums"], raw["counts"], sigma=sigma)
    return {**raw, **scores}