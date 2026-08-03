from functools import partial
from multiprocessing import Pool
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm
from scipy.ndimage import shift as ndshift


def simulate_single(
    sample,
    can,
    n_steps,
    use_clamp=True,
    clamp_fraction=0.15,
    clamp_radius_factor=1.5,
):
    """Run one CAN simulation seeded near a target point on the manifold.

    """
    #Iniate activity zero for the whole network
    N = can.connectivity_matrix.shape[0]
    S = np.zeros((N, 1))
    #Compute distance from the sample point to all neurons
    dists = can.manifold.metric(
        sample.reshape(1, -1),
        can.neurons_coordinates,
    ).ravel()

    # Initial activity 
    S[dists <= can.spacing] = 1.0

    #clamping logic
    if use_clamp and clamp_fraction > 0:
        clamp_radius = clamp_radius_factor * can.spacing 
        inside = (dists <= clamp_radius).reshape(-1, 1)
        n_clamped = int(n_steps * clamp_fraction)
        for _ in range(n_clamped):
            S = can.step_stateless(S)
            S *= inside
        n_free = n_steps - n_clamped
    else:
        n_free = n_steps

    S = can.run_stateless(S, n_free)
    return S.ravel()

def simulate_many(
    samples,
    can,
    n_steps,
    use_clamp=True,
    clamp_fraction=0.15,
    clamp_radius_factor=1.5,
    show_progress=True,
):
    """
    Simulate for every sample, returning stacked final states.
    
    """
    iterator = (
        tqdm(samples, desc="Simulating bumps", unit="sample", dynamic_ncols=True)
        if show_progress else samples
    )
    return np.array([
        simulate_single(
            s, can, n_steps,
            use_clamp=use_clamp,
            clamp_fraction=clamp_fraction,
            clamp_radius_factor=clamp_radius_factor,
        )
        for s in iterator
    ])


def extract_bump_coords(final_states, can):
    """Get the bump position from each final state.

    Returns shape (M, 3) array of (theta_1, theta_2, theta_3) coordinates.
    """
    n1, n2, n3 = can.nx(0), can.nx(1), can.nx(2)
    coords = []
    for state in final_states:
        state_3d = state.reshape(n1, n2, n3)
        i1, i2, i3 = np.unravel_index(np.argmax(state_3d), state_3d.shape)
        coords.append([
            can.idx2coord(i1, 0),
            can.idx2coord(i2, 1),
            can.idx2coord(i3, 2),
        ])
    return np.array(coords)



def shift_and_settle(can, ref3d, injected_grid_coords, n_resettle=50):
    """
    For each target phase in injected_grid_coords (grid units),
    shift the reference lattice to that phase, load into the network,
    settle for n_resettle steps, and return the final state.
    """
    M = len(injected_grid_coords)
    final_states = np.empty((M, can.S.size))

    for i, (dx, dy, dz) in enumerate(tqdm(injected_grid_coords, desc="Settling phases")):
        init = ndshift(ref3d, (dx, dy, dz), mode="grid-wrap", order=1).ravel()
        can.S = init.reshape(-1, 1)
        for _ in range(n_resettle):
            can()
        final_states[i] = can.S.ravel()

    return final_states


def recover_phases(final_states, ref3d):
    """
    Recover the lattice phase of each settled state relative to ref3d
    using cross-correlation in Fourier space.
    """
    shape = ref3d.shape
    ref_fft = np.fft.fftn(ref3d)
    recovered = []

    for state in final_states:
        F = np.fft.fftn(state.reshape(shape)) * np.conj(ref_fft)
        xc = np.fft.ifftn(F).real
        shift = np.array(np.unravel_index(np.argmax(xc), shape))
        recovered.append(shift)

    return np.array(recovered)