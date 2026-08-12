from made.can import CAN, relu
from dataclasses import dataclass, field
import numpy as np

def torus_grid(n: int) -> np.ndarray:
    """Coordinates of every neuron on an n x n x n torus lattice."""
    return (np.indices((n, n, n)).reshape(3, -1).T) * (2 * np.pi / n)


def kernel_field_on_grid(kernel, metric, n: int, offset=None,
                         grid: np.ndarray = None) -> np.ndarray:
    """Connection strength from one neuron to every neuron, as an (n, n, n) volume.

    Measures the torus distance from every lattice point to the offset, then
    passes those distances through the kernel``.



    Args:
        kernel: maps distances to weights, e.g. a ``Kernel_BF``.
        metric: called as ``metric(points, centre)``; handles the wrap-around.
        n: neurons per axis.
        offset: where to centre the kernel. Each of the QAN's six CANs shifts it
            along one axis. None means the origin.
        grid: a precomputed ``torus_grid(n)``, if you already have one.
    """
    #Create the grid if it is not already created
    if grid is None:
        grid = torus_grid(n)
    centre = np.zeros((1, 3))
    
    if offset is not None:
        centre[0] = offset
    return kernel(metric(grid, centre).reshape(n, n, n))


def finite_k_peak(kernel, metric, n: int):
    """Eigenvalues of a shift-invariant recurrent kernel on an n^3 torus grid.

    A distance-only kernel is a convolution, so its eigenvalues are the FFT of
    the single connection volume ``kernel_field_on_grid`` returns.

    Returns:
        peak (float): largest FINITE-wavenumber eigenvalue. Must exceed 1 for a
            bump-forming (Turing) instability; ``QAN`` rescales the kernel so this
            equals ``target_margin``.
        dc (float): the uniform (k=0) eigenvalue. Strongly negative for an
            inhibition-dominated DoG; it sets the forward-Euler stability ceiling
            dt/tau < 2 / (1 - dc*alpha).
    """
    What  = np.fft.fftn(kernel_field_on_grid(kernel, metric, n)).real
    dc    = float(What[0, 0, 0])
    Wf    = What.copy(); Wf.flat[0] = -np.inf          # mask the DC bin
    peak  = float(Wf.max())
    return peak, dc

@dataclass(kw_only=True)
class CAN3D(CAN):
    """CAN with tunable feedforward drive b. Might not have much of a difference

    Inherits all behavior from CAN, but change step_stateless to be able to tune b 

    b, build_connectivity, kernel and dt are required: they come from
    NetworkConfig, by way of the QAN that builds this. kw_only=True is what
    allows them to be required at all, since the parent class already defines
    fields that have defaults.

    Attributes:
        b (float): Constant feedforward excitatory drive.
        build_connectivity (bool): dense numpy matrix vs the torch FFT path.
        kernel: the Kernel_BF the QAN passes in.
        dt (float): forward-Euler step, in the same units as tau.
    """
    b: float
    build_connectivity: bool
    kernel: object
    dt: float

    def __post_init__(self):
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
            raise ValueError("NaN values detected in new state.")

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
    """Center-surround Kernel by Burak and Fiete, difference of Gaussians recurrent kernel.

    sigma_e < sigma_i gives a narrow excitatory peak minus a broader inhibitory surround.

    Attributes:
        alpha (float): Overall gain of the kernel (set from target_margin / peak).
        lambda_net (float): Bump spacing wavelength, in radians.
        ratio (float): How much narrower the excitatory Gaussian is than the inhibitory.
    """

    lambda_net: float #How far apart the bumps end up, in radians
    ratio:      float #How much narrower the excitatory Gaussian is than the inhibitory.
    alpha:      float = 1.0 # Overall gain of the whole kernel, scales everything up
                            # or down uniformly. This is for different network sizes.

    sigma_i: float = field(init=False) #Width of the broad, inhibitory Gaussian
    sigma_e: float = field(init=False) # Width of the narrow, excitatory Gaussian

    def __post_init__(self):
        self.sigma_i = self.lambda_net / np.sqrt(6.0)
        self.sigma_e = self.sigma_i / np.sqrt(self.ratio)

    def __call__(self, d):
        """Applies the kernel"""
        d2 = np.asarray(d, dtype=float) ** 2
        k = (np.exp(-d2 / (2 * self.sigma_e ** 2))
                    - np.exp(-d2 / (2 * self.sigma_i ** 2)))
        return self.alpha * k # scale kernel with network size / target_margin