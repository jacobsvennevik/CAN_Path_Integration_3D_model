from dataclasses import dataclass, field
from made.manifolds import AbstractManifold
from made.qan import QAN
from network.CAN3D import CAN3D, Kernel_BF, finite_k_peak
from network import torus3D_manifold
import numpy as np


@dataclass
class Torus3DQAN(QAN):
    """
    QAN for a 3-torus manifold.
    Uses 6 offset CAN3Ds.
    All three angular dimensions are periodic in [0, 2π].
    Inherits behavior from MADE QAN.
    """
    manifold: AbstractManifold = field(
        default_factory=torus3D_manifold.Torus3D
    )
    # --- kernel ---
    spacing:            float = 0.1   # radians between neighboring neurons on the torus (resolution)
    lambda_net:         float = 1.26  # kernel width
    a:                  float = 1.0   # relative amplitude of the excitatory (narrow)
                                      # Gaussian vs the inhibitory (wide) Gaussian 
    ratio:              float = 1.05  # ratio between

    
    target_margin:      float = 1.5   

    # --- dynamics ---
    b:                  float = 0.3   # constant feedforward baseline drive added to every neruon so that the 
                                      # network moves. 
    offset_magnitude:   float = 0.19  # the shift between each CAN pairing
    dt:                 float = 0.5   # forward-Euler step size, in the same units as tau
    velocity_gain:      float = 1.0   # scalar converting movement in the world into drive strength on the six CANs. 
    build_connectivity: bool  = False # If to build the dense matrix or the FFT-based TorchBackend.

    # --- derived, not settable ---
    gain: float = field(init=False, default=0.0) #the  gain of the kernel based on 

    @classmethod
    def from_config(cls, cfg):
        return cls(spacing=cfg.spacing, lambda_net=cfg.lambda_net, a=cfg.a,
                   ratio=cfg.ratio, b=cfg.b, offset_magnitude=cfg.offset_magnitude,
                   target_margin=cfg.target_margin,
                   dt=getattr(cfg, "dt", 0.5),
                   velocity_gain=getattr(cfg, "velocity_gain", 1.0),
                   build_connectivity=cfg.build_connectivity)

    def __post_init__(self):
        """Build one DoG kernel, derive its gain from target_margin, inject it."""
        self.kernel = Kernel_BF(lambda_net=self.lambda_net, ratio=self.ratio, a=self.a, gain=1.0)
        n = int(np.ceil(2 * np.pi / self.spacing))
        peak, _ = finite_k_peak(self.kernel, self.manifold.metric, n)
        if peak <= 0:
            raise ValueError("No finite-k instability (check sigma_e<sigma_i, lambda_net).")
        self.gain = self.target_margin / peak
        self.kernel.gain = self.gain

        self.cans = []
        for d in range(self.manifold.dim):
            for direction in [1, -1]:
                self.cans.append(
                    CAN3D(
                        self.manifold, self.spacing, self.a, self.kernel.sigma_i,   # alpha,sigma slots vestigial
                        build_connectivity=self.build_connectivity, b=self.b,
                        kernel=self.kernel, dt=self.dt,
                        weights_offset=lambda x, d=d, direction=direction: (
                            self.coordinates_offset(x, d, direction, self.offset_magnitude)),
                    )
                )

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        """Offset coordinates along one dimension creaing an asymetric weight matricies, this makes the QAN drift and wrapping modulo 2π to create the periodicity."""
        theta = theta.copy()
        theta[:, dim] += direction * offset_magnitude
        theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta

    def make_trajectory(self, n_steps: int = 1000, max_speed: float = 0.005 / np.sqrt(3)) -> np.ndarray:
        """Test path generation with incoomensurate rates. This means that trajectories never 
        repeats and thefore will cover T^3."""
        t = np.linspace(0, max_speed * n_steps, n_steps)
        traj = np.zeros((n_steps, self.manifold.dim))
        traj[:, 0] = np.mod(t, 2 * np.pi)
        traj[:, 1] = np.mod(np.sqrt(2)*t, 2 * np.pi)
        traj[:, 2] = np.mod(np.sqrt(3)*t, 2 * np.pi)# third incommensurate freq
        return traj

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        """Does boundary correction for the angular velocity. So that when animal is at a boundary the velocity updates are correct"""
        delta = theta - theta_prev
        for d in range(3):
            if delta[d] > np.pi:
                delta[d] -= 2 * np.pi
            elif delta[d] < -np.pi:
                delta[d] += 2 * np.pi
        return delta

    @property
    def velocity_gains(self) -> float:
        return self.velocity_gain * self.cans[0].tau / self.offset_magnitude     # calibrate velocity_gain

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Maps the velocity to the correct CAN pairing."""
        dim = i // 2          # 0,0 → dim 0 | 1,1 → dim 1 | 2,2 → dim 2
        sign = 1 if i % 2 == 0 else -1
        return sign * self.velocity_gains * theta_dot[dim]