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
        
    def generate_trajectory(self):
        """
        Random walk in physical 2D space.
        Returns world_pos (sequence of positions), velocity_body_seq (sequence of speeds), 
        torus_gt (sequence of positions on the torus manifold), scale.
        """
        cfg   = self.config.experiment          
        rng   = np.random.default_rng(cfg.seed) 
        scale = cfg.scale                        


        #Pre-allocate two arrays of zeroes in 3-dimensions
        world_pos  = np.zeros((cfg.n_steps, 3))
        v_body_seq = np.zeros((cfg.n_steps, 3))
        #persistent random walk
        heading = rng.uniform(0, 2 * np.pi) #random heading
        # Reflect at boundaries ±(env_size/2)
        limit = cfg.env_size / 2
        for t in range(1, cfg.n_steps):
            heading += rng.normal(0, 0.1)          # small turn per step, gaussian
            v = cfg.speed * np.array([np.cos(heading), np.sin(heading), 0.0]) #velocity heading at constant speed
            #update world positon
            new_pos = world_pos[t - 1] + v
            for dim in range(2):  # only x, y for 2D arena
                if new_pos[dim] > limit or new_pos[dim] < -limit:
                    heading = np.pi - heading if dim == 0 else -heading  # reflect
                    v = cfg.speed * np.array([np.cos(heading), np.sin(heading), 0.0]) #Recompute the velocity vector using the reflected heading
                    new_pos = world_pos[t - 1] + v 
            world_pos[t] = new_pos
            v_body_seq[t] = v

        return world_pos, v_body_seq, world_to_torus_gt(world_pos, scale)
    
