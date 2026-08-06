from dataclasses import dataclass, field
from made.manifolds import AbstractManifold
from made.qan import QAN
from model.network.CAN3D import CAN3D, Kernel_BF, finite_k_peak
from model.network import torus3D_manifold
from model.metrics import wrapped_angle_diff
import numpy as np


@dataclass(kw_only=True)
class Torus3DQAN(QAN):
    """
    QAN for a 3-torus manifold.
    Uses 6 offset CAN3Ds.
    All three angular dimensions are periodic in [0, 2π].
    Inherits behavior from MADE QAN.

    Every parameter is required. The defaults live in ``config.NetworkConfig``,
    so the usual way to build one is to name only what you are changing::

        Torus3DQAN.from_config(NetworkConfig(spacing=0.3, lambda_net=2.5))

    Constructing directly works too, it just means naming every parameter.
    """
    manifold: AbstractManifold = field(
        default_factory=torus3D_manifold.Torus3D
    )
    # --- kernel ---
    spacing:            float   # radians between neighboring neurons on the torus (resolution)
    lambda_net:         float   # kernel width
    a:                  float   # relative amplitude of the excitatory (narrow)
                                # Gaussian vs the inhibitory (wide) Gaussian
    ratio:              float   # ratio between

    target_margin:      float   # target peak eigenvalue, means how far above the turing thershold 

    # --- dynamics ---
    b:                  float   # constant feedforward baseline drive added to every neruon so that the
                                # network moves.
    offset_magnitude:   float   # the shift between each CAN pairing
    dt:                 float   # forward-Euler step size, in the same units as tau
    velocity_gain:      float   # scalar converting movement in the world into drive strength on the six CANs.
    build_connectivity: bool    # If to build the dense matrix or the FFT-based TorchBackend.

    @classmethod
    def from_config(cls, cfg):
        """Build from a NetworkConfig. The one intended entry point."""
        return cls(spacing=cfg.spacing, lambda_net=cfg.lambda_net, a=cfg.a,
                   ratio=cfg.ratio, b=cfg.b, offset_magnitude=cfg.offset_magnitude,
                   target_margin=cfg.target_margin,
                   dt=cfg.dt, velocity_gain=cfg.velocity_gain,
                   build_connectivity=cfg.build_connectivity)

    def __post_init__(self):
        """Build one DoG kernel, derive its gain from target_margin, inject it."""
        self.kernel = Kernel_BF(lambda_net=self.lambda_net, ratio=self.ratio, a=self.a, gain=1.0)
        n = int(np.ceil(2 * np.pi / self.spacing))
        peak, _ = finite_k_peak(self.kernel, self.manifold.metric, n)
        if peak <= 0:
            raise ValueError("No finite-k instability (check sigma_e<sigma_i, lambda_net).")
        self.kernel.gain = self.target_margin / peak

        # can_dims[i] is the axis CAN i listens to, can_signs[i] its direction.
        # Index i refers to the same CAN in all three lists.
        self.cans = []
        self.can_dims = []
        self.can_signs = []
        for d in range(self.manifold.dim):
            for direction in [1, -1]:
                self.can_dims.append(d)
                self.can_signs.append(float(direction))
                self.cans.append(
                    CAN3D(
                        self.manifold, self.spacing, self.a, self.kernel.sigma_i,   # alpha,sigma slots vestigial
                        build_connectivity=self.build_connectivity, b=self.b,
                        kernel=self.kernel, dt=self.dt,
                        weights_offset=lambda x, d=d, direction=direction: (
                            self.coordinates_offset(x, d, direction, self.offset_magnitude)),
                    )
                )
        self.can_dims = np.array(self.can_dims, dtype=int)
        self.can_signs = np.array(self.can_signs, dtype=float)

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
        return wrapped_angle_diff(theta, theta_prev)

    def theta_dot_at(self, trajectory: np.ndarray, t: int) -> np.ndarray:
        """Angular velocity at step t of a wrapped trajectory, zero at t = 0.
        """
        if t == 0:
            return np.zeros(trajectory.shape[1], dtype=np.float32)
        return self.compute_theta_dot(
            trajectory[t].copy(), trajectory[t - 1].copy()
        ).astype(np.float32)

    @property
    def drive_per_theta_dot(self) -> float:
        """v_m per unit θ̇ on each CAN axis (derived from velocity_gain)."""
        return self.velocity_gain * self.cans[0].tau / self.offset_magnitude

    def can_velocity_drives(self, theta_dot: np.ndarray) -> np.ndarray:
        """Per-CAN velocity drive v_m for an angular velocity, shape (n_cans,).

        Each CAN is driven by the velocity component along its own axis, signed
        by its own direction. The torch backend computes the same thing as
        tensors, from ``can_dims``/``can_signs``.
        """
        theta_dot = np.asarray(theta_dot, dtype=float)
        return self.can_signs * self.drive_per_theta_dot * theta_dot[self.can_dims]