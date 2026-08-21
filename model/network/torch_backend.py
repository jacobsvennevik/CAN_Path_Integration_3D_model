"""
Runner for the torch backend, in larger simulations need to use this class.
For ligther runs just use the MADE framework and the QAN3D.py and CAN3D classes.
"""

import gc
import numpy as np
import torch
from dataclasses import dataclass, field

from model.metrics import wrapped_angle_diff
from model.network.CAN3D import kernel_field_on_grid, torus_grid


class BumpTracker:
    """Follows one bump on the n³ torus by local centre of mass.

    Seeded at a start coordinate, find the nearest bump centre by using Argmax.
    Then advance the tracker to the next frame by finding the local centre of mass. 
    It is a bit of a simple tracker, so visualisation should be used to see that the bump beeing tracked
    isnt switching.
    """

    def __init__(self, n: int, radius: int = 4, seed_radius: int = None):
        self.n = int(n) #neurons pr axis
        self.radius = int(radius) #spherical tracking window
        self.seed_radius = int(max(self.radius, n // 8) if seed_radius is None
                               else seed_radius) #inital seed window

        #Holds the relative coordinates of the cells in the tracking window
        off = np.arange(-self.radius, self.radius + 1) #offset grid
        self._OX, self._OY, self._OZ = np.meshgrid(off, off, off, indexing="ij")
        self._sphere_mask = (
            (self._OX * self._OX + self._OY * self._OY + self._OZ * self._OZ)
            <= self.radius * self.radius
        )

        self._soff = np.arange(-self.seed_radius, self.seed_radius + 1)
        self._SX, self._SY, self._SZ = np.meshgrid(
            self._soff, self._soff, self._soff, indexing="ij")

        self.c_prev = None   # bump centre in grid cell
        self.pos = None      # unwrapped position in radians
        self.history = [] # recoding the path of the tracker, for visualisation

    def _local_com(self, S_3d: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Centre of mass of the tracking window around cell c, takes all of the activity weighted by lenth from cell c"""
        n = self.n
        ci = np.round(c).astype(int) % n #Round to an integer cell, wrap into range.
        # Only the (2*radius+1)^3 window is upcast, not the whole volume.
        w = S_3d[(ci[0] + self._OX) % n, (ci[1] + self._OY) % n, #the window of activity, % n is the wrapping of the torus
                 (ci[2] + self._OZ) % n].astype(np.float64)
        np.maximum(w, 0.0, out=w) #Clip negatives
        w *= self._sphere_mask  # sphere only

        wsum = w.sum()
        if wsum < 1e-12:
            return c   
        #takes the sum of all of the activity * the position and weights it by the activity of all neurons, 
        # see methology (x.x) for equation                                   
        com = np.array([(self._OX * w).sum(), (self._OY * w).sum(),
                        (self._OZ * w).sum()]) / wsum
        return (ci + com) % n                             # sub-cell centre

    def seed(self, S_3d: np.ndarray, theta_0: np.ndarray) -> np.ndarray:
        """Lock onto the strongest bump near theta_0 and start counting there."""
        n = self.n #number of neurons
        theta_0 = np.asarray(theta_0, dtype=np.float64) #starting cell

        c0 = np.round(theta_0 / (2 * np.pi) * n).astype(int) % n #cell window corner
        win = S_3d[(c0[0] + self._SX) % n, (c0[1] + self._SY) % n, #The wide window, and the most active cell in it
                   (c0[2] + self._SZ) % n]  
        s0 = np.unravel_index(int(np.argmax(win)), win.shape) #s0 indexes into the window arrary. _soff converts back to an offset
        c_seed = ((c0 + np.array([self._soff[s0[0]], self._soff[s0[1]], #Add the weighet avergae to the argmax cell to refine.
                                  self._soff[s0[2]]])) % n).astype(np.float64)

        #update the previous cell to the new cell
        self.c_prev = self._local_com(S_3d, c_seed) 
        self.pos = theta_0 % (2 * np.pi) #accumulate the position
        return self.pos.copy()

    def advance(self, S_3d: np.ndarray) -> np.ndarray:
        """Follow the bump into one new frame and return the unwrapped position."""
        n = self.n
        c_new = self._local_com(S_3d, self.c_prev) #find the new cell
        d = wrapped_angle_diff(c_new, self.c_prev, period=n)   # smallest torus step, the distance
        self.pos = self.pos + d / n * (2 * np.pi) #accumulate the position
        self.c_prev = c_new #update the center cell
        return self.pos.copy()

    def advance_frames(self, volumes: np.ndarray) -> np.ndarray:
        """Decode a (T, n, n, n) stack; tracker state carries across frames/chunks."""
        out = np.empty((len(volumes), 3), dtype=np.float64)
        for t, frame in enumerate(volumes):
            out[t] = self.advance(frame)
        return out


@dataclass
class TorchBackend:
    """
    Torch-backed simulation engine for a Torus3DQAN.
    """

    qan: object                                  # Torus3DQAN instance
    torch_dtype: torch.dtype = torch.float32    #Uses cheeper float32 

    device: torch.device = field(init=False)
    S:      torch.Tensor = field(init=False) #The current neural activity
    coords: torch.Tensor = field(init=False) #Torus coordinates
    tau:    torch.Tensor = field(init=False) #Neural time constant
    dt:     torch.Tensor = field(init=False) #Integration step
    b: torch.Tensor = field(init=False) #bias term to produce activity in every neuron
    tracker: BumpTracker = field(init=False, default=None) #persistent bump follower

    def __post_init__(self):
        self.device = self._get_torch_device()
        self._init_torch_backend()

    # I use mps
    def _get_torch_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        
    def _compute_fft_kernels(self, n: int) -> torch.Tensor:
        """Pre-FFT the recurrent kernel for each of the 6 offset CANs.
        Returns W_fft, shape (6, n, n, n//2+1), complex64."""
        metric     = self.qan.manifold.metric
        kernel_fn  = self.qan.kernel              # the injected Kernel_BF (your B&F DoG)
        offset_mag = self.qan.offset_magnitude

        grid = torus_grid(n)                      # built once, reused per offset

        fft_device = torch.device("cpu") if self.device.type == "mps" else self.device
        n_cans = len(self.qan.cans)
        W_fft = torch.empty((n_cans, n, n, n // 2 + 1), dtype=torch.complex64,
                            device=fft_device)

        for i, (dim, sign) in enumerate(zip(self.qan.can_dims, self.qan.can_signs)):
            # constant axis offset δ for this CAN (flat-torus Killing field)
            delta = np.zeros(3); delta[dim] = sign * offset_mag

            kernel = kernel_field_on_grid(kernel_fn, metric, n, offset=delta, grid=grid)

            W_fft[i] = torch.fft.rfftn(
                torch.as_tensor(kernel, dtype=torch.float32, device=fft_device),
                dim=(-3, -2, -1),
            )
        return W_fft
 
 
    def _apply_W_fft(self, S_shared: torch.Tensor) -> torch.Tensor:
        n = self.n
        N = n ** 3

        S_3d = S_shared[0].squeeze(-1).reshape(n, n, n)

        # Remove the .cpu() calls — stay on device for CUDA
        if self.device.type == "mps":
            S_3d  = S_3d.cpu()
            W_fft = self.W_fft.cpu()
        else:
            W_fft = self.W_fft.to(self.device)  # already there after init, no-op

        S_fft  = torch.fft.rfftn(S_3d, dim=(-3, -2, -1))
        Ws_fft = W_fft * S_fft.unsqueeze(0)
        Ws_3d  = torch.fft.irfftn(Ws_fft, s=(n, n, n), dim=(-3, -2, -1))

        return Ws_3d.reshape(6, N, 1).to(self.device)

    def _init_torch_backend(self):
        """
        Starts the torch backend. Takes the NumPy CAN created through the MADE framework.
        Moves the parts we can make fast into torch, to not blow up memory we remove dense
        duplicate matrices.
        """
        cans = self.qan.cans

        # count CANs (should be 6) and neurons
        n_cans = len(cans)
        N = cans[0].S.shape[0]
        self.n = int(round(N ** (1/3)))    
        self.W_fft = self._compute_fft_kernels(self.n)

        # empty space for activity states (current activity of all neurons)
        self.S = torch.empty(
            (n_cans, N, 1),
            dtype=self.torch_dtype,
            device=self.device,
        )
        
        # Copies one CAN at the time, converts into float32, then copies into Torch tensor
        for i, can in enumerate(cans):
            S_i = can.S.astype(np.float32, copy=False)

            self.S[i].copy_(
                torch.as_tensor(
                    S_i,
                    dtype=self.torch_dtype,
                    device=self.device,
                )
            )
            

            # After W_i is copied to torch, we no longer keep the dense NumPy matrices.
            if hasattr(can, "connectivity_matrix"):
                del can.connectivity_matrix
            del S_i
            # Forces python to clean up
            gc.collect()

        self.S = self.S.contiguous()

        # Go through each parameter used during simulation, make them into Torch tensors
        self.tau = torch.tensor(
            cans[0].tau,
            dtype=self.torch_dtype,
            device=self.device,
        )

        self.dt = torch.tensor(
            cans[0].dt,
            dtype=self.torch_dtype,
            device=self.device,
        )

        self.b = torch.tensor(
            self.qan.b,
            dtype=self.torch_dtype,
            device=self.device,
        )
        # Coordinates of each neuron
        self.coords = torch.tensor(
            cans[0].neurons_coordinates.astype(np.float32),
            dtype=self.torch_dtype,
            device=self.device,
        )
        # Which axis each CAN listens to, and with which sign, in the order
        # self.qan.cans is built.
        self.dims_torch = torch.tensor(
            self.qan.can_dims,
            dtype=torch.long,
            device=self.device,
        )
        self.signs_torch = torch.tensor(
            self.qan.can_signs,
            dtype=self.torch_dtype,
            device=self.device,
        )

        gc.collect()

        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
            

    def reset(self, theta_0: np.ndarray, radius: float = 0.05):
        """
        Reset all CANs to the starting trajectory point.
        The neural bump is then at this point.
        """
        theta = torch.as_tensor(
            theta_0.reshape(1, -1), #inital position on the torus
            dtype=self.torch_dtype,
            device=self.device,
        )

        coords = self.coords  # shape (N, 3)

        # shortest signed difference between every neuron and theta_0, wrapped
        diff = wrapped_angle_diff(coords.unsqueeze(0), theta.unsqueeze(1))

        #torus distance from neuron i to theta_0 in one distance instead of 3D
        distances = torch.linalg.norm(diff, dim=-1).squeeze(0)  # shape (N,)

        #How large the initial bump should be
        effective_radius = torch.max(distances) * radius
        
        #Initiate an all zero acitivty state, one CAN activity vector.
        S0 = torch.zeros(
            (self.coords.shape[0], 1),
            dtype=self.torch_dtype,
            device=self.device,
        )
        #Turn on neurons close to the starting coordinate, at attractor amplitude b.
        S0[distances <= effective_radius] = self.qan.b

        #Copy that same starting bump into all six CANs.
        self.S = S0.unsqueeze(0).expand(len(self.qan.cans), -1, -1).clone()

    def step(self, theta_dot: np.ndarray):
        """Compute S_tot internally and step."""
        S_tot = torch.mean(self.S, dim=0) #average activity acrosse CAN
        return self.step_from_shared_state(S_tot, theta_dot)

    def step_from_shared_state(self, S_tot: torch.Tensor, theta_dot: np.ndarray,
        check_nan: bool = False,
    ) -> torch.Tensor:
        """
        Implements the bump updates driven by the velcoity using the shared mean neural state, the current neural activity S_tot. 
        In this way we take a step, and the bump moves.
        """
        N = S_tot.shape[0]

        #The current neural activity, gives us where the bump is currently, the mean field
        S_shared = S_tot.unsqueeze(0).expand(6, N, 1)
        #Apply each CANs shifted weight matricies W to the shared bump of activity       
        Ws = self._apply_W_fft(S_shared)
        #The pr CAN velocity inut
        td  = torch.as_tensor(theta_dot, dtype=self.torch_dtype, device=self.device)  # (3,)
        v_m  = (self.signs_torch * self.qan.drive_per_theta_dot * td[self.dims_torch]).view(6, 1, 1)
        # Apply the recurrent input to each neuron, 
        # passed through a relu and shifted up by the bias b and velocity v_m.
        drives = torch.relu(Ws + self.b + v_m)               
   
        #Updates the current neural state
        self.S = self.S + (self.dt / self.tau) * (drives - self.S)

        if check_nan and torch.isnan(self.S).any().item():
            raise ValueError("NaN values detected in torch QAN state.")

        return self.S

    
    def current_volume(self) -> np.ndarray:
        """Mean-field activity as an (n, n, n) float32 numpy array."""
        n = self.n
        return (self.S.mean(dim=0).squeeze().detach().cpu().numpy()
                .reshape(n, n, n))

    def dominant_k(self):
        """Nyquist-folded wavevector of the peak Fourier mode, and its |k|."""
        n = self.n
        vol = self.current_volume()
        F = np.abs(np.fft.fftn(vol)); F.flat[0] = 0.0
        kk = np.arange(n)
        kk = np.where(kk <= n // 2, kk, kk - n)
        i = np.unravel_index(int(F.argmax()), F.shape)
        k = np.array([kk[i[0]], kk[i[1]], kk[i[2]]], dtype=float)
        return k, float(np.linalg.norm(k))

    def bump_period_cells(self) -> float:
        """Lattice period in cells, from the dominant Fourier mode of the current state."""
        _, kmag = self.dominant_k()
        return self.n / max(kmag, 1.0)

    def seed_tracker(self, theta_0, radius=None, seed_radius=None):
        """Start a persistent bump follower on the CURRENT state.

        Default radius is period/4 (as before). The COM window is a sphere.
        """
        period = self.bump_period_cells()
        if radius is None:
            radius = max(1, int(round(period / 4)))
        if seed_radius is None:
            seed_radius = max(radius + 1, int(round(period / 2)))
        n = self.n
        vol = self.current_volume()
        self.tracker = BumpTracker(n, radius=int(radius), seed_radius=int(seed_radius))
        return self.tracker.seed(vol, theta_0)

    def form_lattice(self, theta_0, settle=3000, min_peakedness=0.0):
        """Reset, run a fixed undriven settle, record peakedness. Does not drive."""
        theta_0 = np.asarray(theta_0, dtype=np.float64).copy()
        self.reset(theta_0, radius=0.05)
        zero_v = np.zeros(3, dtype=np.float32)
        for _ in range(int(settle)):
            self.step_from_shared_state(torch.mean(self.S, dim=0), zero_v)
        s = torch.mean(self.S, dim=0); m = float(s.mean())
        pk = float(s.max()) / m if m > 1e-12 else float("nan")
        self.last_peakedness = pk
        if not np.isfinite(pk) or pk < min_peakedness:
            print(f"WARNING: peakedness {pk:.2f} after {int(settle)} settle steps "
                  f"(want >= {min_peakedness:.1f}; ~1 = no lattice)")
        return pk

    def drive(self, trajectory: np.ndarray,
              return_states=False, radius: int = None,
              seed_radius: int = None,
              display_stride: int = 8,
              snapshot_stride: int = 100,
              on_volume=None) -> np.ndarray:
        """Decode while driving along ``trajectory``. Lattice must already be formed.

        ``on_volume(t, vol)`` is optional. Called once with ``t=-1`` on the
        latched volume before the first step, then with ``t=0..T-1`` after
        each step. Diagnostic scripts use this; production callers ignore it.
        """
        theta_0 = trajectory[0, :].copy()
        self.seed_tracker(theta_0, radius=radius, seed_radius=seed_radius)

        n, T = self.n, trajectory.shape[0]
        if on_volume is not None:
            on_volume(-1, self.current_volume())

        pos = np.empty((T, 3), dtype=np.float64)
        self.display_stride = display_stride
        self.display_marginals = np.empty(
            ((T + display_stride - 1) // display_stride, 3, n),
            dtype=np.float32,
        )
        self.snapshot_stride = snapshot_stride
        self.snapshot_times = np.arange(0, T, snapshot_stride)
        n_snap = len(self.snapshot_times)
        self.snapshots = np.empty((n_snap, n, n, n), dtype=np.float32)
        self.snapshot_cells = np.empty((n_snap, 3), dtype=np.float64)
        buf = (torch.empty((T, self.S.shape[1]), dtype=self.torch_dtype, device=self.device)
               if return_states else None)

        for t in range(T):
            self.step_from_shared_state(
                torch.mean(self.S, dim=0),
                self.qan.theta_dot_at(trajectory, t),
            )
            S_tot = torch.mean(self.S, dim=0).squeeze()
            v = S_tot.reshape(n, n, n)
            vol = v.detach().cpu().numpy()

            pos[t] = self.tracker.advance(vol)
            if on_volume is not None:
                on_volume(t, vol)

            if t % display_stride == 0:
                self.display_marginals[t // display_stride] = torch.stack(
                    [v.sum(dim=(1, 2)), v.sum(dim=(0, 2)), v.sum(dim=(0, 1))]
                ).detach().cpu().numpy()
            if t % snapshot_stride == 0:
                si = t // snapshot_stride
                self.snapshots[si] = vol
                self.snapshot_cells[si] = self.tracker.c_prev
            if buf is not None:
                buf[t] = S_tot

        self.pos_com_unwrapped = pos
        out = pos % (2 * np.pi)
        if return_states:
            return out, buf.detach().cpu().numpy()
        return out

    def simulate(self, trajectory: np.ndarray, settle=3000,
                 return_states=False, radius: int = None,
                 seed_radius: int = None,
                 min_peakedness: float = 3.0,
                 display_stride: int = 8,
                 snapshot_stride: int = 100) -> np.ndarray:
        """Settle then drive. Prefer form_lattice + drive when gates run in between."""
        self.form_lattice(trajectory[0], settle=settle, min_peakedness=min_peakedness)
        return self.drive(trajectory, return_states=return_states, radius=radius,
                          seed_radius=seed_radius, display_stride=display_stride,
                          snapshot_stride=snapshot_stride)

    def run(self, trajectory: np.ndarray):
        """
        Run torch dynamics without decoding.

        Useful when you only care about final CAN state.
        """

        theta_0 = trajectory[0, :].copy()
        self.reset(theta_0, radius=0.05)

        for t in range(trajectory.shape[0]):
            self.step(self.qan.theta_dot_at(trajectory, t))

        return self.S

    def sync_to_cans(self):
        """
        Copy torch states back into the individual CAN3D NumPy objects.
        """
        S_np = self.S.detach().cpu().numpy()

        for i, can in enumerate(self.qan.cans):
            can.S = S_np[i]

    def get_states(self):
        """
        Return current torch states as a NumPy array with shape (6, N, 1).
        """
        return self.S.detach().cpu().numpy()
    
    def allocate_state_buffer(self, T: int, stride: int = 1) -> "torch.Tensor":
        """Make room an on-device buffer for recording S_tot at each timestep."""
        N = self.S.shape[1]
        n_frames = (T + stride - 1) // stride
        return torch.empty(
            (n_frames, N),
            dtype=self.torch_dtype,
            device="cpu",
        )

    def record_state_to_buffer(self, buf: "torch.Tensor", t: int, stride: int = 1) -> None:
        """Write current S_tot into row t of the buffer. No CPU transfer."""
        buf[t // stride] = self.S.mean(dim=0).squeeze().to(buf.device)

    def buffer_to_numpy(self, buf: "torch.Tensor") -> "np.ndarray":
        """Single CPU transfer of the full buffer."""
        return buf.cpu().numpy()

    def allocate_ratemap(self, total_bins: int, sub_t=None) -> tuple:
        """On-device accumulator for the 2-D per-neuron rate map.."""
        N = len(sub_t) if sub_t is not None else self.S.shape[1]
        sums   = torch.zeros((total_bins, N), dtype=torch.float32, device=self.device)
        counts = torch.zeros( total_bins,      dtype=torch.float32, device=self.device)
        return sums, counts

    def record_ratemap(self, acc: tuple, flat_bin: int, sub_t=None) -> None:
        """Accumulate current S_tot into bins (much cheeper and quicker).
        The caller computes flat_bin so the backend stays unaware of arena geometry."""
        sums, counts = acc
        with torch.no_grad():
            s = self.S.mean(dim=0).squeeze()
            if sub_t is not None:
                s = s[sub_t]
            sums[flat_bin] += s
        counts[flat_bin] += 1.0

    def ratemap_to_numpy(self, acc: tuple, bins: int, ndim: int = 2) -> tuple:
        """Single CPU transfer from ratemap to numpy"""
        sums, counts = acc
        shape_s = (bins,) * ndim + (-1,)
        shape_c = (bins,) * ndim
        return (sums.cpu().numpy().reshape(shape_s),
                counts.cpu().numpy().reshape(shape_c))
    
    def allocate_shuffle_ratemap(self, total_bins: int, n_neurons: int,
                                 n_shuffle: int):
        """On-device (n_shuffle, total_bins, n_neurons) buffer for time-shifted
        sums used by the spatial-info / sparsity Z-score pipeline."""
        return torch.zeros((n_shuffle, total_bins, n_neurons),
                           dtype=torch.float32, device=self.device)
 
    def record_shuffle_ratemap(self, shuf_sums, flat_indices, t: int,
                               lags, sub_t=None) -> None:
        """Accumulate time-shifted activity for each shuffle."""
        with torch.no_grad():
            s = self.S.mean(dim=0).squeeze()
            if sub_t is not None:
                s = s[sub_t]
            T = len(flat_indices)
            for j in range(shuf_sums.shape[0]):
                b = int(flat_indices[(t + int(lags[j])) % T])
                shuf_sums[j, b] += s

