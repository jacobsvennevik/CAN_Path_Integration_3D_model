from made.manifolds import AbstractManifold, ParameterSpace, Range
from made.metrics import Metric, PeriodicEuclidean
from dataclasses import dataclass, field
import numpy as np


class ParameterSpace3D(ParameterSpace):
    """Extends ParameterSpace with 3D."""

    def sample(self, n: int, pads: list[float] =
               None) -> np.ndarray:
        """
        Returns points sampled from the parameter space.
        Used for visualisation.
        For 3D: returns n points as an (n^3, 1) array
        For everything else falls back to the parrent class

        Args:
            n (int): Number of points to sample
            pads (list[float]): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of sampled points
        """
        if pads is None:
            pads = [0.0] * self.dim
        if self.dim == 3:
            #My simple addition
            assert (
                len(pads) == 3
            ), f"Incorrect number of pasd for manifold dimension: {self.pads}"
            # Create meshgrid
            x = self.ranges[0].sample(n, pads[0]) # n points along theta_1
            y = self.ranges[1].sample(n, pads[1]) # n points along theta_2
            z = self.ranges[2].sample(n, pads[2]) # n points along theta_3
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij") #indexing="ij" imoprtant for visualization.
            # Return as (n*m*o, 3) array
            return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        else:
            # Fall back to the parent class for 1D and 2D
            return super().sample(n, pads)

    def sample_with_spacing(
        self, spacing: float, pads: list[float] = None
    ) -> np.ndarray:
        """
        Returns points sampled from the parameter space with a fixed spacing.
        Used for neuron placement in the CAN. 
        For 3D: returns n points as an (n^3, 1) array
        For everything else falls back to the parrent class

        Args:
            spacing (float): Fixed spacing between points
            pads (list[float]): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of sampled points
        """
        if pads is None:
            pads = [0.0] * self.dim

        if self.dim == 3:
            assert (
                len(pads) == 3
            ), "Incorrect number of pads for manifold dimension"
            # Calculate number of points needed in each dimension by looping over each range defined in the parameterspace. 
            range_sizes = [r.end - r.start for r in self.ranges]
            # number of neurons per dimension
            ns = [int(np.ceil(size / spacing)) for size in range_sizes]
            # Create meshgrid of 3D coordinates, so theta_i of each neuron.
            x = self.ranges[0].sample(ns[0], pads[0])
            y = self.ranges[1].sample(ns[1], pads[1])
            z = self.ranges[2].sample(ns[2], pads[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            # Return as (n*m*o, 3) array
            return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        else:
            #If other dimension use parent function
            return super().sample_with_spacing(spacing, pads) 
        
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
    
    


