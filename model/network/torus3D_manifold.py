from made.manifolds import AbstractManifold, ParameterSpace, Range
from made.metrics import Metric, PeriodicEuclidean
from dataclasses import dataclass, field
import numpy as np


class ParameterSpace3D(ParameterSpace):
    """Extends ParameterSpace with 3D."""

    def _meshgrid_columns(self, per_axis_counts: list[int],
                          pads: list[float]) -> np.ndarray:
        """Places points evenly along the axises of the manifold and combines them into (theta_1, theta_2, theta_3) coordinates. 
        That is where each neuron sits.
        """
        assert len(pads) == 3, (
            f"Incorrect number of pads for manifold dimension: got {len(pads)}, "
            f"expected 3")
        axes = [r.sample(count, pad)
                for r, count, pad in zip(self.ranges, per_axis_counts, pads)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    def sample(self, n: int, pads: list[float] =
               None) -> np.ndarray:
        """
        Returns points sampled from the parameter space.
        Used for visualisation.
        For 3D: returns n^3 points as an (n^3, 3) array
        For everything else falls back to the parrent class

        Args:
            n (int): Number of points to sample per axis
            pads (list[float]): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of sampled points
        """
        if pads is None:
            pads = [0.0] * self.dim
        if self.dim != 3:
            # Fall back to the parent class for 1D and 2D
            return super().sample(n, pads)
        return self._meshgrid_columns([n] * 3, pads)

    def sample_with_spacing(
        self, spacing: float, pads: list[float] = None
    ) -> np.ndarray:
        """
        Returns points sampled from the parameter space with a fixed spacing.
        Used for neuron placement in the CAN. 
        For 3D: returns an (n^3, 3) array, n set by the spacing
        For everything else falls back to the parrent class

        Args:
            spacing (float): Fixed spacing between points
            pads (list[float]): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of sampled points
        """
        if pads is None:
            pads = [0.0] * self.dim
        if self.dim != 3:
            #If other dimension use parent function
            return super().sample_with_spacing(spacing, pads)
        # neurons per dimension, from each range's own extent
        counts = [int(np.ceil((r.end - r.start) / spacing)) for r in self.ranges]
        return self._meshgrid_columns(counts, pads)


@dataclass
class Torus3D(AbstractManifold):
    """
    New manifold structure, 3D manifold representing a three dimensional torus T^3.

    All dimensions are periodic, representing the three possible axises of movement.
    Topology: T^3 = S^1 x S^1 x S^1.
    """
    dim: int = 3
    parameter_space: ParameterSpace3D = field(
        default_factory=lambda: ParameterSpace3D([
        # Three periodic dimensions (3D), each with period 2*pi.
        # Points that wrap around, they are periodic.
        # giving the topology of a 3-torus T^3
        Range(0, 2 * np.pi, periodic=True), # [0, L1]
        Range(0, 2 * np.pi, periodic=True), # [0, L2]
        Range(0, 2 * np.pi, periodic=True), # [0, L3]
        ])
    )
    # Computes shortest wrap-around distance between two points for all dimensions.
    metric: Metric = field(
        default_factory=lambda: PeriodicEuclidean(dim=3, periodic=[True, True, True])
    )
    
    


