import numpy as np
import copy
from typing import Optional
import torch

from model.plane_estimation import (
    BinghamDistribution,
    predict,
    update,
    uniform_prior,
)
from scipy.spatial.transform import Rotation as Rot
from model.network.torch_backend import TorchBackend

def build_rotation_matrix(n_hat: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Returns R such that R @ n_hat = z_hat, via the closed-form
    vector-to-vector rotation (Rodrigues). 
    """
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    z_hat = np.array([0.0, 0.0, 1.0])

    v = np.cross(n_hat, z_hat)   # rotation axis * sin(theta)
    c = float(np.dot(n_hat, z_hat))  # cos(theta)

    # already aligned -> identity (this is the common case on a flat floor)
    if c > 1.0 - 1e-10:
        return np.eye(3)
    # anti-aligned (n_hat points straight down) -> 180° flip
    if c < -1.0 + 1e-10:
        return np.diag([1.0, -1.0, -1.0])

    vx = np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0 ],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

def pi_star(v_alloc: np.ndarray) -> np.ndarray:
    """
    Applies the pushforward π★ (differential of π) to an allocentric velocity vector.
    It is the indetity matrix so a trivial computasion,
    in our case but kept for consistency vis a vi the MADE framework (Claudi et, al. 2025)
    """
    return np.asarray(v_alloc, dtype=float)

def step_filter(current, displacement, kappa, rho):
    """Single predict and update cycle."""
    predicted = predict(current, rho)
    return update(predicted, displacement, kappa)


class PathIntegrator:
    """
    Path integrator coupling the Bingham plane filter with the
    T³ QAN.
    """
    def __init__(self, qan, kappa=10.0, rho=0.999, scale=1.0, 
                 initial_estimate=None, record_stride=10,
                 plane_mode="bayesian", true_n_hat = None,
                 decode_chunk=4096, decode_radius=4, decode_seed_radius=None):
        self.qan = qan 
        self.kappa = kappa #likelihood consentration for the Bingham update.
        self.rho = rho # Bingham concentration decay ρ
        self.scale = scale
        self.backend = TorchBackend(qan) 
        self._bingham_state = initial_estimate or uniform_prior() #starting belief for n̂
        self._theta = np.zeros(qan.manifold.dim) #decoded position
        self._theta_0 = np.zeros(qan.manifold.dim) #where the bump was seeded
        self._n_hat_corrected = None  # gravity-disambiguated, stored on self
        self.plane_mode = plane_mode #whether to use the true plane mode or the Bingham mode
        if true_n_hat is None:
            true_n_hat = np.array([0.0, 0.0, 1.0])      # gravity / flat floor
        true_n_hat = np.asarray(true_n_hat, dtype=float)
        self._true_n_hat = true_n_hat / np.linalg.norm(true_n_hat)
        self.decode_chunk = int(decode_chunk) #how many steps to buffer on-device before decoding in one go
        # Bump-tracker window sizes, in grid cells. The tracking window has to stay
        # below half the bump spacing, or the centre of mass straddles two bumps and
        # the decoded velocity comes out wrong-signed.
        self.decode_radius = decode_radius
        self.decode_seed_radius = decode_seed_radius
        self.history = {
            "n_hat": [], #MAP plane normal at each step
            "z1": [], "z2": [], #concentration parameters
            "v_body": [], #body velocity
            "v_alloc": [], #allocentric velocity after rotation
            "target_speed_rad": [], #push-forward velocity fed to CANs
            "theta": [] #decoded CAN position
            } 
        # Optional recording buffers
        # S_tot_buffer: stays on-device (no per-step CPU transfer)
        self.S_tot_buffer      = None 
        self.bingham_snapshots = None
        self.record_stride = record_stride #run() records the state every Nth step
        self.ratemap_sums   = None   # set by run(..., ratemap_bins=N) when > 0
        self.ratemap_counts = None   
        
    def warmup(self, n_steps: int = 100):
        """Let the lattice settle and the filter deflate, then lock the tracker on.

        The tracker is re-seeded at the end rather than followed through warmup:
        while the pattern is still forming, the bump can move further in one step
        than the tracking window is wide, which loses it. Nothing has moved during
        warmup, so the position afterwards is still the seed position.
        """
        zero_v = np.zeros(3)
        for _ in range(n_steps):
            self.backend.step(zero_v)
            self._bingham_state = predict(self._bingham_state, self.rho)
        self._theta = self._seed_tracker()

    def _seed_tracker(self) -> np.ndarray:
        return self.backend.seed_tracker(
            self._theta_0, radius=self.decode_radius,
            seed_radius=self.decode_seed_radius)

    def _advance(self, v_body, g_hat):
        """The per-step physics shared by step() and run(). No decode, no history."""
        v_body = np.asarray(v_body, dtype=float)
        
        d_norm = np.linalg.norm(v_body)
        
        if d_norm > 1e-9:
            #we only want direction, not magnitude
            v_body_t_unit = v_body / d_norm
            # run the bingham filter
            self._bingham_state = step_filter(self._bingham_state, v_body_t_unit, self.kappa, self.rho)

        #plane mode either bayesian or true plane mode
        if self.plane_mode == "true":
            n_hat = self._true_n_hat
        else:
            # Bayesian: extract MAP estimate, disambiguate sign with gravity
            n_hat = self._bingham_state.M[:, -1]
            if np.dot(n_hat, g_hat) > 0:
                n_hat = -n_hat
            
        self._n_hat_corrected = n_hat

        #build rotation matrix and rotate velocity 
        R = build_rotation_matrix(n_hat, g_hat)
        v_alloc = R @ v_body #allocentric velocity
    
        target_speed_rad = v_alloc * self.scale  # rad/step (history key; not the backend unit)

        # backend.step expects rad per unit TIME (theta_dot_at's convention)
        self.backend.step(target_speed_rad / self.qan.dt)
        return n_hat, v_alloc, target_speed_rad

    def step(self, v_body: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Advance the integrator by one timestep.

        """
        g_hat = np.asarray(g, dtype=float)
        g_hat = g_hat / np.linalg.norm(g_hat)
        n_hat, v_alloc, target_speed_rad = self._advance(v_body, g_hat)
        # decode current position
        self._theta = self._decode_from_torch()

        # Record into history
        self.history["n_hat"].append(n_hat.copy())
        self.history["z1"].append(self._bingham_state.z1)
        self.history["z2"].append(self._bingham_state.z2)
        self.history["v_body"].append(v_body.copy())
        self.history["v_alloc"].append(v_alloc.copy())
        self.history["target_speed_rad"].append(target_speed_rad .copy())
        self.history["theta"].append(self._theta.copy())

        return self._theta


    def run(self, v_body_sequence: np.ndarray, g: np.ndarray, record: bool = False,
            flat_indices: np.ndarray = None, ratemap_bins: int = 0,
            ratemap_ndim: int = 2, sub_idx: np.ndarray = None,
            n_shuffle: int = 0, lags: np.ndarray = None) -> np.ndarray:
        """
        Run the full pipeline over a pre-computed velocity sequence.

        Same job as calling step() in a loop, but we skip the per-step decode
        (that .cpu() call syncs the GPU every step) and skip the per-step list
        appends (those fragment RAM on long runs). Instead we stash the bump
        state on-device and decode a chunk at a time, and write history straight
        into pre-allocated arrays. step() itself is left alone since the scoring
        code drives the network through it.
        """

        T = v_body_sequence.shape[0] #total timesteps
        dim = self.qan.manifold.dim
        dev = self.backend.device
        N   = self.backend.S.shape[1]
        theta_history = np.zeros((T, dim)) #place to store decoded positions

        # history goes into flat arrays now, not growing lists
        _h_n_hat   = np.empty((T, 3),   dtype=np.float64)
        _h_z1      = np.empty(T,         dtype=np.float64)
        _h_z2      = np.empty(T,         dtype=np.float64)
        _h_v_body  = np.empty((T, 3),   dtype=np.float64)
        _h_v_alloc = np.empty((T, 3),   dtype=np.float64)
        _h_tsr     = np.empty((T, dim), dtype=np.float64)

        # small on-device buffer, decode + dump to CPU once it fills (keeps memory bounded)
        chunk = min(self.decode_chunk, T)
        _S_chunk = torch.empty((chunk, N), dtype=torch.float32, device=dev)

        if record:
            _buf  = self.backend.allocate_state_buffer(T, stride=self.record_stride)
            _bing = []
        else:
            _buf  = None
            _bing = None
        
                # neuron subsample → torch index tensor
        sub_t = (torch.tensor(sub_idx, dtype=torch.long, device=self.backend.device)
                 if sub_idx is not None else None)
        n_neurons = len(sub_idx) if sub_idx is not None else self.backend.S.shape[1]
            
        # rate-map accumulator (2-D or 3-D)
        _acc = None
        if flat_indices is not None and ratemap_bins > 0:
            total_bins = ratemap_bins ** ratemap_ndim
            _acc = self.backend.allocate_ratemap(total_bins, sub_t)
        
        _shuf = None
        if n_shuffle > 0 and lags is not None and _acc is not None:
            _shuf = self.backend.allocate_shuffle_ratemap(total_bins, n_neurons, n_shuffle)

        # g is fixed for the whole run, so normalise it once not every step
        g = np.asarray(g, dtype=float)
        g_hat = g / np.linalg.norm(g)

        chunk_start = 0 #first step currently held in _S_chunk

        for t in range(T):            
            v_body = np.asarray(v_body_sequence[t], dtype=float)
            n_hat, v_alloc, target_speed_rad = self._advance(v_body, g_hat)
            # stash the bump state on-device, no .cpu() here
            _S_chunk[t - chunk_start] = self.backend.S.mean(dim=0).squeeze()

            # write history straight into the arrays
            _h_n_hat[t]   = n_hat
            _h_z1[t]      = self._bingham_state.z1
            _h_z2[t]      = self._bingham_state.z2
            _h_v_body[t]  = v_body
            _h_v_alloc[t] = v_alloc
            _h_tsr[t]     = target_speed_rad

            if _acc is not None:
                self.backend.record_ratemap(_acc, int(flat_indices[t]), sub_t)
            if _shuf is not None:
                self.backend.record_shuffle_ratemap(_shuf, flat_indices, t, lags, sub_t)
            if record and (t % self.record_stride== 0):
                self.backend.record_state_to_buffer(_buf, t, stride=self.record_stride)
                _bing.append(copy.deepcopy(self._bingham_state))

            # chunk full (or last step) -> decode it all at once and dump to CPU.
            # The tracker keeps its state between calls, so chunks join up.
            filled = t - chunk_start + 1
            if filled == chunk or t == T - 1:
                theta_history[chunk_start:t + 1] = self.backend.track_batch(_S_chunk[:filled])
                chunk_start = t + 1

        self._theta = theta_history[-1] #last decoded position, same as before

        # hand history back as the same dict-of-lists the rest of the code reads
        self.history["n_hat"]            = list(_h_n_hat)
        self.history["z1"]               = list(_h_z1)
        self.history["z2"]               = list(_h_z2)
        self.history["v_body"]           = list(_h_v_body)
        self.history["v_alloc"]          = list(_h_v_alloc)
        self.history["target_speed_rad"] = list(_h_tsr)
        self.history["theta"]            = list(theta_history)

        if record:
            self.S_tot_buffer      = self.backend.buffer_to_numpy(_buf)
            self.bingham_snapshots = _bing
        else:
            self.S_tot_buffer      = None
            self.bingham_snapshots = None
        
        if _acc is not None:
            self.ratemap_sums, self.ratemap_counts = \
                self.backend.ratemap_to_numpy(_acc, ratemap_bins, ndim=ratemap_ndim)
        else:
            self.ratemap_sums = self.ratemap_counts = None

        if _shuf is not None:
            shape = (n_shuffle,) + (ratemap_bins,) * ratemap_ndim + (-1,)
            self.ratemap_shuf_sums = _shuf.cpu().numpy().reshape(shape)
        else:
            self.ratemap_shuf_sums = None

        return theta_history

    def _decode_from_torch(self) -> np.ndarray:
        """Current bump position, unwrapped, by advancing the backend's tracker.

        One call = one frame, so this must be called exactly once per network
        step. run() bypasses it and drives the tracker in chunks instead.
        """
        return self.backend.track_step()

    def concentration_eigenvalue_gap(self) -> float:
        """
        Diagnostics for uncertinity of the filter.
        Large gap, filter is confident.
        Small gap, still uncertain between two candidate axes.
        """
        return self._bingham_state.z2 - self._bingham_state.z1

    def reset(self, theta_0: np.ndarray, initial_estimate: Optional[BinghamDistribution] = None):
        """
        Reset filter and CAN states without rebuilding the full object.
        For running multiple trials with the same QAN hyperparameters.
        """
        if initial_estimate is None:
            self._bingham_state = uniform_prior()
        else:
            self._bingham_state = initial_estimate

        self.backend.reset(theta_0)
        self._theta_0 = np.asarray(theta_0, dtype=np.float64).copy()
        # Seeded here so the integrator is usable straight after reset(); warmup()
        # re-seeds on the settled lattice, which is the lock that matters.
        self._theta = self._seed_tracker()

        for key in self.history:
            self.history[key] = []
        
        self.S_tot_buffer      = None
        self.bingham_snapshots = None