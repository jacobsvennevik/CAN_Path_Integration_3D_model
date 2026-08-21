import numpy as np
from config import ExperimentConfig
from experiments.base import BaseExperiment, world_to_torus_gt

class Arena2DConfig(ExperimentConfig):
    @property
    def run_name(self) -> str:
        return f"arena2d_T{self.n_steps}_kap{self.kappa}_seed{self.seed}_envSize{self.env_size:}_gridSpacing{self.grid_spacing}"
    

class Arena2DExperiment(BaseExperiment):

    condition_label = "arena_2d"
    ratemap_ndim          = 2        
    ratemap_n_sub         = 300      # blind random draw from the full N
    ratemap_seed          = 0        # independent of cfg.seed; for reproducible subsample
    ratemap_n_shuffle     = 50       # <-- ADD: enables circular-shift Z (sinfo_z/sidx_z)
    ratemap_active_thresh = 1e-3     # inherited default, restated for visibility
        
    def generate_trajectory(self, turn_std: float = None, n_steps=None, seed=None):
        """
        Random walk in physical 2D space.
        Returns world_pos (sequence of positions), velocity_body_seq (sequence of speeds), 
        torus_gt (sequence of positions on the torus manifold).
        """
        cfg   = self.config.experiment
        n_steps = cfg.n_steps if n_steps is None else int(n_steps)
        seed = cfg.seed if seed is None else int(seed)
        rng   = np.random.default_rng(seed)
        scale = cfg.scale
        dt = self.config.network.dt
        torus_inc = cfg.target_speed_rad_per_time * dt   # rad/step
        world_speed = torus_inc / cfg.scale              # m/step
        omega_std = cfg.omega_std if turn_std is None else float(turn_std)
        step_turn_std = omega_std * np.sqrt(dt)


        #Pre-allocate two arrays of zeroes in 3-dimensions
        world_pos  = np.zeros((n_steps, 3))
        v_body_seq = np.zeros((n_steps, 3))
        #persistent random walk
        heading = rng.uniform(0, 2 * np.pi) #random heading
        # Reflect at boundaries ±(env_size/2)
        limit = cfg.env_size / 2
        for t in range(1, n_steps):
            heading += rng.normal(0, step_turn_std)     # Wiener heading: ω_std √dt
            v = world_speed * np.array([np.cos(heading), np.sin(heading), 0.0]) #velocity heading at constant speed
            #update world positon
            new_pos = world_pos[t - 1] + v
            for dim in range(2):  # only x, y for 2D arena
                if new_pos[dim] > limit or new_pos[dim] < -limit:
                    heading = np.pi - heading if dim == 0 else -heading  # reflect
                    v = world_speed * np.array([np.cos(heading), np.sin(heading), 0.0]) #Recompute the velocity vector using the reflected heading
                    new_pos = world_pos[t - 1] + v 
            world_pos[t] = new_pos
            v_body_seq[t] = v

        return world_pos, v_body_seq, world_to_torus_gt(world_pos, scale)
    
