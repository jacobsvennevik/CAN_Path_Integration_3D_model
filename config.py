from dataclasses import dataclass, field
import numpy as np

@dataclass
class NetworkConfig:
    """
    Sets up the network for the experiment: Network + Movement + Integration also a flag realted to memory
    """
    spacing:            float = 2.0 * np.pi / 64  # 64 cells/axis (2^6); sheet size is 2π/spacing
    lambda_net:         float = 1.0  # ~10.2 cells/period at n=64; 0.6932 would be 7.1 and fail the cells/period gate
    ratio:              float = 1.05  # B&F gamma/beta
    alpha:              float = 0.15  # loop gain ~1.7 at this lambda and n
    
    #movement of the bump
    b:                  float = 1.0 #Positive global exitasion to the whole network
    offset_magnitude:   float = 0.073 # ~9% of the 0.785 rad lattice period; 0.067–0.078 window

    #integration
    dt:                 float = 0.25 #forward-Euler step size, same units as tau
    tau:                float = 5.0  # neural time constant; dt/tau ≤ 0.125 at tau=2
    velocity_gain:      float = 5.0 # b * tau seed (tau=5); fit_gain refines
    
    #flag related to building dense numoy matricies or skipping that
    build_connectivity: bool  = False


@dataclass
class ExperimentConfig:
    """Sets up the enviorment for the experiment: Environment + Bingham filter."""
    env_size:         float = 2.0    # metres: size of the box the animal walks in (walk boundaries)
    n_steps:          int   = 3000 #Defult timesteps
    seed:             int   = 0
    kappa:            float = 10.0 #Bingham filter measurement strength:
    rho:              float = 0.999  # Bingham concentration decay ρ (not the Optuna drive composite θ̇τ/δ)
    grid_spacing:     float = 0.48    # metres per full 2π wrap = the torus period
    target_speed_rad_per_time: float = 0.002  # desired bump speed, rad per unit TIME (not per step)
    # Heading diffusion. Units rad · time^{-1/2}, not rad/time.
    # Per-step Gaussian is omega_std * sqrt(dt) so heading variance grows in time, not steps.
    # Correlation time τ_h = 2 / ω² = 2222 time units at 0.03 (≈ 8900 steps at dt=0.25).
    omega_std:        float = 0.03
    record_stride: int = 20 #How many recordings
    ratemap_bins: int = 40
    scale:            float = field(init=False)
    
    def __post_init__(self):
        self.scale = (2 * np.pi) / self.grid_spacing      # the metres→radians conversion from the world manifold to the tours manifold and visa versa
    def m_to_rad(self, x_m):   return x_m * self.scale
    def rad_to_m(self, x_rad): return x_rad / self.scale


@dataclass
class AnalysisConfig:
    """Offline scoring parameters, read by everything under analysis/."""
    bins:            int   = 40     # histogram bins/axis for rate map + autocorrelogram
    smooth_sigma:    float = 1.75   # gaussian_filter sigma, in BINS (see note below)
    autocorr_th:     float = 0.1    # autocorrelation zeroing threshold
    n_neuron:        int   = 300    # default cells subsampled by score_run
    go_az_precision: int   = 48     # global-order azimuth samples
    go_al_precision: int   = 24     # global-order altitude samples

@dataclass
class RunConfig:
    network:    NetworkConfig    = field(default_factory=NetworkConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    # no spacing/grid_spacing assert: different unit systems, no reason to match
    
def world_to_normalized(world_pos, env_size):
    """Map physical position (metres, within ±env_size/2) to the [-1,1] cube
    that every scorer assumes. Arena-anchored: the ruler is the wall, not the
    trajectory, so grid SPACING stays physically meaningful across runs."""
    half = env_size / 2.0
    return np.clip(world_pos / half, -1.0, 1.0)

def world_to_flat_bins(world_pos, env_size, bins, ndim=2):
    """Answers which flat bin does each position fall in.

    The rate-map accumulator is a flat array of length bins**ndim, so each
    position has to collapse to a single integer. Bin numbers along each axis are
    combined as digits in base `bins`: a 3-D position in bins (i, j, k) maps to
    i*bins**2 + j*bins + k. That is NumPy's own row-major order, so
    reshape((bins,) * ndim) at the end puts every count in the right cell.

    Only the leading `ndim` columns are used, which is how the 2-D arena drops z
    from its (T, 3) positions.
    """
    x = world_to_normalized(world_pos[:, :ndim], env_size)  # shared normalize; index-clip below handles bounds
    idx = np.clip(np.floor((x + 1.0) * 0.5 * bins).astype(np.int64), 0, bins - 1)
    flat = idx[:, 0]
    for d in range(1, ndim):
        flat = flat * bins + idx[:, d]
    return flat

def world_to_flat_bins_3d(world_pos, env_size, bins):
    """The 3-D case, for call sites that ask for it by name."""
    return world_to_flat_bins(world_pos, env_size, bins, ndim=3)


