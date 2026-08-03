"""
Runner for the torch backend, in larger simulations need to use this class.
For ligther runs just use the MADE framework and the QAN3D.py and CAN3D classes.
"""

import gc
import numpy as np
import math
import torch
from dataclasses import dataclass, field


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
    linear_drive: bool = False #test of thanh

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

        # torus grid in [0, 2π)³, shape (n³, 3)
        theta_grid = (np.indices((n, n, n)).reshape(3, -1).T) * (2 * np.pi / n)

        fft_device = torch.device("cpu") if self.device.type == "mps" else self.device
        W_fft = torch.empty((6, n, n, n // 2 + 1), dtype=torch.complex64, device=fft_device)

        for i in range(6):
            dim       = i // 2
            direction = 1.0 if i % 2 == 0 else -1.0

            # constant axis offset δ for this CAN (flat-torus Killing field)
            delta = np.zeros((1, 3)); delta[0, dim] = direction * offset_mag

            # distance from every grid point to the offset centre, on the torus metric
            dist   = metric(theta_grid, delta).reshape(n, n, n)

            # connectivity kernel — whatever is injected (Kernel_BF = B&F DoG)
            kernel = kernel_fn(dist)

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
                #dimension index each CAN listens to: 0,0 | 1,1 | 2,2
        self.dims_torch = torch.tensor(
            [0, 0, 1, 1, 2, 2],
            dtype=torch.long,
            device=self.device,
        )
        #the +/- offset direction sign paired with dims_torch (even CAN +, odd CAN -)
        self.signs_torch = torch.tensor(
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
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

        # compute difference between every neuron and theta_+
        diff = coords.unsqueeze(0) - theta.unsqueeze(1)
        diff = (diff + np.pi) % (2 * np.pi) - np.pi #wrapping differences insidce [0, 2π]

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
        #Turn on neurons close to the starting coordinate.
        S0[distances <= effective_radius] = 1.0

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
        v_m  = (self.signs_torch * self.qan.velocity_gains * td[self.dims_torch]).view(6, 1, 1)
        # Apply the recurrent input to each neuron, 
        # passed through a relu and shifted up by the bias b and velocity v_m.
        drives = torch.relu(Ws + self.b + v_m)               
   
        #Updates the current neural state
        self.S = self.S + (self.dt / self.tau) * (drives - self.S)

        if check_nan and torch.isnan(self.S).any().item():
            raise ValueError("NaN values detected in torch QAN state.")

        return self.S

    
    def decode_flow_batch(self, S_chunk: torch.Tensor, theta_0: np.ndarray,
                      radius: int = 4, seed_radius: int = None) -> np.ndarray:
        """
        Follow one bump by local centre-of-mass and integrate its displacement.
    
        Seeded at theta_0 -- reset() places the bump there, which for theta_0 = 0
        is the grid CORNER, not the grid centre the old code searched.
    
        Args:
            radius:      half-width of the tracking window, in grid cells. Keep
                        well below the lattice period so it follows one bump.
            seed_radius: half-width of the initial search window. After settling
                        the nearest bump need not sit exactly on theta_0, so
                        search wider than you track. Defaults to n // 8.
        """
        n = self.n
        M = S_chunk.shape[0]
    
        # Zero-copy view when S_chunk is already a CPU float32 tensor.
        S = S_chunk.detach().cpu().numpy().reshape(M, n, n, n)
    
        off = np.arange(-radius, radius + 1)
        OX, OY, OZ = np.meshgrid(off, off, off, indexing="ij")
    
        if seed_radius is None:
            seed_radius = max(radius, n // 8)
        soff = np.arange(-seed_radius, seed_radius + 1)
        SX, SY, SZ = np.meshgrid(soff, soff, soff, indexing="ij")
    
        c0 = (np.round(np.asarray(theta_0, dtype=np.float64) / (2 * np.pi) * n)
            .astype(int)) % n
        win = S[0][(c0[0] + SX) % n, (c0[1] + SY) % n, (c0[2] + SZ) % n]
        s0 = np.unravel_index(int(np.argmax(win)), win.shape)
        c_prev = ((c0 + np.array([soff[s0[0]], soff[s0[1]], soff[s0[2]]])) % n
                ).astype(np.float64)
    
        pos = np.zeros((M, 3))
        pos[0] = np.asarray(theta_0, dtype=np.float64) % (2 * np.pi)
    
        for t in range(M):
            ci = (np.round(c_prev).astype(int)) % n
            # Only the (2*radius+1)^3 window is upcast, not the whole buffer.
            w = S[t][(ci[0] + OX) % n, (ci[1] + OY) % n,
                    (ci[2] + OZ) % n].astype(np.float64)
            np.maximum(w, 0.0, out=w)
    
            wsum = w.sum()
            if wsum < 1e-12:
                c_new = c_prev                                   # bump faded: hold
            else:
                com = np.array([(OX * w).sum(), (OY * w).sum(), (OZ * w).sum()]) / wsum
                c_new = (ci + com) % n                           # sub-neuron centre
    
            if t > 0:
                d = c_new - c_prev
                d = (d + n / 2) % n - n / 2                      # smallest torus step
                pos[t] = pos[t - 1] + d / n * (2 * np.pi)
            c_prev = c_new
    
        return pos % (2 * np.pi)

    def simulate(self, trajectory: np.ndarray, decode: str = "phase", settle=300, return_states = False) -> np.ndarray:
        """
        Simulate feeding a generated trajectory into the network.
        Returning a decoded trajectory of the bump position at each timestep.
        """

        theta_0 = trajectory[0, :].copy()
        self.reset(theta_0, radius=0.05)  # Puts the bump at the initial seed position
        
        zero_v = np.zeros(3, dtype=np.float32)
        for _ in range(int(settle)):
            self.step_from_shared_state(torch.mean(self.S, dim=0), zero_v)
 

        # If using FFT phase decoding, store the current bump pattern as the
        # reference anchor.
        if decode == "phase":
            self.capture_reference(theta_0)

        T = trajectory.shape[0]  # total timesteps
        N = self.S.shape[1]      # neurons per CAN

        # On-device buffer for the mean-field state at every timestep.
        buf = torch.empty((T, N), dtype=self.torch_dtype, device=self.device)

        # Velocity computation over each timestep
        for t, theta in enumerate(trajectory):
            if t == 0:
                theta_dot = np.zeros(theta.shape, dtype=np.float32)
            else:
                theta_dot = (
                    self.qan.compute_theta_dot(  # computes the displacement between the current position theta and the previous position
                        theta.copy(),
                        trajectory[t - 1, :].copy(),
                    )
                ).astype(np.float32)

            # Compute the average activity of the CANs (cancels out the asymmetry
            S_tot = torch.mean(self.S, dim=0)

            # Update activity state based on the average activity of previous step
            self.step_from_shared_state(S_tot, theta_dot)
            
            # Record after step mean-field state.
            buf[t] = torch.mean(self.S, dim=0).squeeze()
            # Burak & Fiete single-bump follower, velocity integrated.
            out = self.decode_flow_batch(buf, theta_0)

        if return_states: #check for notebook 3
            return out, buf.detach().cpu().numpy()
        return out

    def run(self, trajectory: np.ndarray):
        """
        Run torch dynamics without decoding.

        Useful when you only care about final CAN state.
        """

        theta_0 = trajectory[0, :].copy()
        self.reset(theta_0, radius=0.05)

        for t, theta in enumerate(trajectory):
            if t == 0:
                theta_dot = np.zeros(theta.shape, dtype=np.float32)
            else:
                theta_dot = (
                    self.qan.compute_theta_dot(
                        theta.copy(),
                        trajectory[t - 1, :].copy(),
                    )
                ).astype(np.float32)

            self.step(theta_dot)

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
                
