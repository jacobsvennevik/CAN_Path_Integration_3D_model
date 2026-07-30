import numpy as np
from dataclasses import asdict
import numpy as np
from dataclasses import asdict
from config import ExperimentConfig
from experiments.base import BaseExperiment

class Arena3DConfig(ExperimentConfig):
    @property
    def run_name(self) -> str:
        return f"arena3d_T{self.n_steps}_kap{self.kappa}_seed{self.seed}_envSize{self.env_size:}_gridSpacing{self.grid_spacing}"
    
    
class Arena3DExperiment(BaseExperiment):
    condition_label = "arena_3d"
    ratemap_ndim      = 3
    ratemap_n_sub     = 300
    ratemap_n_shuffle = 20
    
    def __init__(self, config, record=True, plane_mode="true"):
        super().__init__(config, record)                  # QAN + integrator_kwargs from Arena2D
        self.integrator_kwargs["plane_mode"] = plane_mode  # flip the filter on/off
        

    def generate_trajectory(self, turn_std: float = 0.1):
        cfg   = self.config.experiment
        rng   = np.random.default_rng(cfg.seed)
        scale = cfg.scale
        limit = cfg.env_size / 2
 
        world_pos  = np.zeros((cfg.n_steps, 3))
        v_body_seq = np.zeros((cfg.n_steps, 3))
 
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)            # random initial unit heading
 
        for t in range(1, cfg.n_steps):
            direction = direction + rng.normal(0, turn_std, size=3)
            direction /= np.linalg.norm(direction)        # diffuse heading on the sphere
            v = cfg.speed * direction
            new_pos = world_pos[t - 1] + v
            for dim in range(3):                          # reflect in x, y AND z
                if new_pos[dim] > limit or new_pos[dim] < -limit:
                    direction[dim] = -direction[dim]      # specular bounce
                    v = cfg.speed * direction
                    new_pos = world_pos[t - 1] + v
            world_pos[t]  = new_pos
            v_body_seq[t] = v
        
        world_scaled = world_pos * scale                                 # applies 3×3 B_inv.T to all dims
        torus_gt = (np.pi + world_scaled) % (2 * np.pi)  # z is columnar not periodic TODO: watch out for this
        return world_pos, v_body_seq, torus_gt
    
    