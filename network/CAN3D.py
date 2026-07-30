from made.can import CAN, relu
from dataclasses import dataclass
import numpy as np

def finite_k_peak(kernel, metric, n: int):
    """Eigenvalues of a shift-invariant recurrent kernel on an n^3 torus grid.
 
    A kernel that depends only on distance is a convolution operator, so its
    eigenvalues are just the FFT of one row (the kernel sampled from the origin).
    This replaces the external ``instability_check`` dependency and reuses the
    same FFT idea as the torch backend's ``_compute_fft_kernels``.
 
    Returns:
        peak (float): largest FINITE-wavenumber eigenvalue. Must exceed 1 for a
            bump-forming (Turing) instability; ``QAN`` rescales the kernel so this
            equals ``target_margin``.
        dc (float): the uniform (k=0) eigenvalue. Strongly negative for an
            inhibition-dominated DoG; it sets the forward-Euler stability ceiling
            dt/tau < 2 / (1 - dc*gain).
    """
    theta_grid = (np.indices((n, n, n)).reshape(3, -1).T) * (2 * np.pi / n)
    dist  = metric(theta_grid, np.zeros((1, 3))).reshape(n, n, n)
    field = kernel(dist)
    What  = np.fft.fftn(field).real
    dc    = float(What[0, 0, 0])
    Wf    = What.copy(); Wf.flat[0] = -np.inf          # mask the DC bin
    peak  = float(Wf.max())
    return peak, dc

@dataclass
class CAN3D(CAN):
    """CAN with tunable feedforward drive b. Might not have much of a difference

    Inherits all behavior from CAN, but change step_stateless to be able to tune b 

    Attributes:
        b (float): Constant feedforward excitatory drive.
        build_connectivity: bool = True dense vs torch built matrix memory 
    """
    b: float = 1.0
    build_connectivity: bool = True
    kernel: object = None        # injected Kernel_BF from the QAN (single source)
    dt: float = 0.5              # explicit forward-Euler step

    def __post_init__(self):
        if self.kernel is None:
            raise ValueError("CAN3D needs an injected kernel (Kernel_BF) from the QAN.")
        self.neurons_coordinates = (
            self.manifold.parameter_space.sample_with_spacing(self.spacing)
        )
        if self.build_connectivity:
            distances = self.manifold.metric.pairwise_distances(
                self.neurons_coordinates, weights_offset=self.weights_offset)
            self.connectivity_matrix = self.kernel(distances)
        self.S = np.zeros((self.neurons_coordinates.shape[0], 1))
    def step_stateless(self, S, u=0):
        """
        Override function

        """
        if not hasattr(self, "connectivity_matrix"):
            raise AttributeError(
                "step_stateless needs connectivity_matrix, which was skipped "
                "(build_connectivity=False). Use the torch FFT backend for fine-spacing runs."
            )
        S_dot = self.connectivity_matrix @ S + u + self.b
        new_S = S + (self.dt / self.tau) * (relu(S_dot) - S)

        if np.any(np.isnan(new_S)):
            raise ValueError(f"NaN values detected in new state.")

        return new_S
    
    @property
    def weight_matrix(self) -> np.ndarray:
        if not hasattr(self, "connectivity_matrix"):
            raise AttributeError(
                "connectivity_matrix was not built (build_connectivity=False). "
                "Use the torch FFT backend (TorchBackend), which never needs the dense matrix."
            )
        return self.connectivity_matrix
    
@dataclass
class Kernel_BF:
    """Center-surround Kernel by Burak and fiete, difference of Gaussians recurrent kernel.

    sigma_e < sigma_i: means 

    Attributes:
        alpha (float): Scaling factor for the kernel
        sigma (float): Width parameter of the Gaussian
    """

    lambda_net: float #How far apart the bumps end up, in radians
    ratio:      float #How much narrower the excitatory Gaussian is than the inhibitory.
    a:          float = 1.0 # Height of the narrow excitatory Gaussian, relative to the broad 
                            # inhibitory Gaussian's fixed height of 1.
    gain:       float = 1.0 # Overall volume knob for the whole kernel, scales everything up
                            # or down uniformly. This is for different network sizes. 

    sigma_i: float = field(init=False) #Width of the broad, inhibitory Gaussian
    sigma_e: float = field(init=False) # Width of the narrow, excitatory Gaussian

    def __post_init__(self):
        self.sigma_i = self.lambda_net / np.sqrt(6.0)
        self.sigma_e = self.sigma_i / np.sqrt(self.ratio)

    def __call__(self, d):
        """Applies ther kernel"""
        d2 = np.asarray(d, dtype=float) ** 2
        k = (self.a * np.exp(-d2 / (2 * self.sigma_e ** 2))
                    - np.exp(-d2 / (2 * self.sigma_i ** 2)))
        return self.gain * k #apply the gain to the kernel, scaling the kernel depending on the network size