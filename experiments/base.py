import os
import json
import pickle
from dataclasses import dataclass, asdict

import numpy as np

from model.metrics import wrapped_angle_diff
from model.path_integration import PathIntegrator
from config import world_to_flat_bins
from model.network.QAN3D import Torus3DQAN


def world_to_torus_gt(world_pos, scale):
    """Ground-truth torus coordinates for a world path, in radians.

    The metres-to-radians conversion is a plain scaling. PeriodicEuclidean lays
    the torus axes directly on the world axes, so there is no basis change to
    apply. 

    The + pi centres the path in the middle of the torus rather than at the
    corner, so a walk starting at the world origin does not sit on the wrap.
    """
    return (np.pi + np.asarray(world_pos) * scale) % (2 * np.pi)


@dataclass
class ExperimentResult:
    world_pos:         np.ndarray
    v_body_seq:        np.ndarray
    torus_gt:          np.ndarray
    theta_hist:        np.ndarray
    n_hat_hist:        np.ndarray
    gap_hist:          np.ndarray
    S_tot_buffer:      np.ndarray   # or None
    bingham_snapshots: list         # or None
    norm_error:        np.ndarray
    mean_norm_error:   float
    condition:         str
    params:            dict
    ratemap_sums:      np.ndarray = None   
    ratemap_counts:    np.ndarray = None
    sub_idx: np.ndarray = None   # neuron subsample indices (3-D path)
    shuf_sums:         np.ndarray = None   # (n_shuffle, bins, ..., n_sub) shuffle maps
    active_mask:       np.ndarray = None   # (n_sub,) bool — neurons with non-trivial activity


class BaseExperiment:
    condition_label = "base" #overwritten by subclasses
    ratemap_ndim          = 2
    ratemap_n_sub         = 0
    ratemap_n_shuffle     = 0
    ratemap_seed          = 0
    ratemap_active_thresh = 1e-3
    
    def __init__(self, config, record=True, plane_mode="bayesian"):
        self.qan = Torus3DQAN.from_config(config.network)
        N_neurons    = self.qan.cans[0].S.shape[0]
        decode_chunk = max(64, int(256e6 / (4 * N_neurons)))
        self.integrator_kwargs = dict(
            kappa=config.experiment.kappa,
            rho=config.experiment.rho,   # Bingham concentration decay ρ, not kernel alpha
            scale=config.experiment.scale,
            plane_mode=plane_mode,                   # "bayesian" (default) or "true"
            decode_chunk=decode_chunk,
        )
        self.config        = config
        self.record        = record
        self.record_stride = config.experiment.record_stride
        
        e = self.config.experiment
        print(
            f"[{self.condition_label}] "
            f"neurons {self.qan.cans[0].S.shape[0]:,} "
            f"arena {e.env_size} m | grid_spacing {e.grid_spacing} m | "
            f"tilings {e.env_size / e.grid_spacing:.2f}"
        )

        
    def generate_trajectory(self, turn_std: float = 0.1, n_steps=None, seed=None):
        """Required hook: return (world_pos, v_body_seq, torus_gt).

        n_steps and seed default to config.experiment when omitted.
        """
        raise NotImplementedError("Should be implemented by subclass")
    
    def run(self, world_pos, v_body_seq, torus_gt, g_vec) -> ExperimentResult:
        bins     = self.config.experiment.ratemap_bins
        env_size = self.config.experiment.env_size
        ndim     = self.ratemap_ndim

        flat = world_to_flat_bins(world_pos, env_size, bins, ndim=ndim)
        
        integrator = PathIntegrator(qan=self.qan, **self.integrator_kwargs)
        integrator.reset(torus_gt[0])
        integrator.warmup(n_steps=100)   # let CAN stabilize, filter converge
        
        # neuron subsample + shuffle lags (one RNG, sequential draws)
        N   = integrator.backend.S.shape[1]
        rng = np.random.default_rng(self.ratemap_seed)

        sub_idx = None
        if self.ratemap_n_sub > 0:
            sub_idx = np.sort(rng.choice(N, size=min(self.ratemap_n_sub, N),
                                         replace=False))

        lags = None
        n_shuffle = self.ratemap_n_shuffle
        if n_shuffle > 0:
            T       = len(world_pos)
            min_lag = max(1, int(T * 0.1))
            if min_lag < T:
                lags = rng.integers(min_lag, T, size=n_shuffle)
            else:
                n_shuffle = 0          # trajectory too short for shuffles
        
        theta_hist = integrator.run(
            v_body_seq, g_vec,
            record=self.record,
            flat_indices=flat,
            ratemap_bins=bins,
            ratemap_ndim=ndim,
            sub_idx=sub_idx,
            n_shuffle=n_shuffle if lags is not None else 0,
            lags=lags,
        )

        gap = np.array(integrator.history["z2"]) - np.array(integrator.history["z1"])
        n_hat_hist = np.array(integrator.history["n_hat"])
        norm_error = self._made_metric(theta_hist, torus_gt, world_pos=world_pos)
        
        active_mask = None
        if sub_idx is not None and integrator.ratemap_sums is not None and integrator.ratemap_counts is not None:
            sums_flat   = integrator.ratemap_sums.reshape(-1, integrator.ratemap_sums.shape[-1])  # (total_bins, n_sub)
            counts_flat = integrator.ratemap_counts.reshape(-1)                                    # (total_bins,)
            valid       = counts_flat > 0                                                          # unvisited bins → nan
            rate_flat   = np.full_like(sums_flat, np.nan)
            rate_flat[valid] = sums_flat[valid] / counts_flat[valid, np.newaxis]
            span        = np.nanmax(rate_flat, axis=0) - np.nanmin(rate_flat, axis=0)
            active_mask = span > self.ratemap_active_thresh * (np.nanmax(span) + 1e-12)

        self.last_integrator = integrator
        #Mostly doing online now so might not need this below anymore TODO
        return ExperimentResult(
            world_pos=world_pos,
            v_body_seq=v_body_seq,
            ratemap_sums=integrator.ratemap_sums,
            ratemap_counts=integrator.ratemap_counts,
            torus_gt=torus_gt,
            theta_hist=theta_hist,
            n_hat_hist=n_hat_hist,
            gap_hist=gap,
            S_tot_buffer=integrator.S_tot_buffer,
            bingham_snapshots=integrator.bingham_snapshots,
            norm_error=norm_error,
            mean_norm_error=float(norm_error[1:].mean()),
            condition="",
            params={},
            sub_idx=sub_idx,
            shuf_sums=integrator.ratemap_shuf_sums,
            active_mask=active_mask,
        )
        

    def save(self, result: ExperimentResult, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        stride = (len(result.world_pos) // result.S_tot_buffer.shape[0]
              if hasattr(result, "S_tot_buffer") and result.S_tot_buffer is not None
              else 1)
        arrays = {
            "world_pos":     result.world_pos[::stride],   # strided to match S_tot_buffer
            "torus_gt":      result.torus_gt[::stride],
            "v_body_seq":    result.v_body_seq,
            "theta_hist":    result.theta_hist,
            "n_hat_hist":    result.n_hat_hist,
            "gap_hist":      result.gap_hist,
            "norm_error":    result.norm_error,
            "record_stride": np.int32(stride),
        }
        if result.S_tot_buffer is not None:
            arrays["S_tot_buffer"] = result.S_tot_buffer   # only added when recording
        np.savez(path, **arrays)

        # persist the full config so the run is reproducible
        with open(path.replace(".npz", "_config.json"), "w") as f:
            json.dump({"condition": result.condition, "params": result.params},
                      f, indent=2)

        if result.bingham_snapshots is not None:
            with open(path.replace(".npz", "_bingham.pkl"), "wb") as f:
                pickle.dump(result.bingham_snapshots, f)

    @staticmethod
    def _made_metric(decoded, ground_truth, world_pos=None):
        if world_pos is not None:
            steps = np.linalg.norm(np.diff(world_pos, axis=0), axis=1)  # metres
        else:
            steps = np.linalg.norm(
                wrapped_angle_diff(ground_truth[1:], ground_truth[:-1]), axis=1
            )
        traj_length = np.cumsum(steps)
        err = np.linalg.norm(wrapped_angle_diff(decoded, ground_truth), axis=1)
        norm_error = err[1:] / (traj_length + 1e-9)
        return np.concatenate([[0.0], norm_error])
    
    def run_experiment(self, g):
        world_pos, v_body_seq, torus_gt = self.generate_trajectory()
        result = self.run(world_pos, v_body_seq, torus_gt, g)
        result.condition = self.condition_label
        result.params = asdict(self.config)
        return result
    