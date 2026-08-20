import numpy as np
from config import ExperimentConfig
from experiments.base import BaseExperiment, world_to_torus_gt

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
        

    def generate_trajectory(self, turn_std: float = 0.1, n_steps=None, seed=None):
        """Independent 3D reflecting walk. n_steps and seed default to config."""
        cfg   = self.config.experiment
        n_steps = cfg.n_steps if n_steps is None else int(n_steps)
        seed = cfg.seed if seed is None else int(seed)
        rng   = np.random.default_rng(seed)
        scale = cfg.scale
        limit = cfg.env_size / 2
        dt = self.config.network.dt
        torus_inc = cfg.target_speed_rad_per_time * dt   # rad/step
        world_speed = torus_inc / cfg.scale              # m/step
 
        world_pos  = np.zeros((n_steps, 3))
        v_body_seq = np.zeros((n_steps, 3))
 
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)            # random initial unit heading
 
        for t in range(1, n_steps):
            direction = direction + rng.normal(0, turn_std, size=3)
            direction /= np.linalg.norm(direction)        # diffuse heading on the sphere
            v = world_speed * direction
            new_pos = world_pos[t - 1] + v
            for dim in range(3):                          # reflect in x, y AND z
                if new_pos[dim] > limit or new_pos[dim] < -limit:
                    direction[dim] = -direction[dim]      # specular bounce
                    v = world_speed * direction
                    new_pos = world_pos[t - 1] + v
            world_pos[t]  = new_pos
            v_body_seq[t] = v

        # NOTE: z is treated as periodic here like x and y. If the lattice ends up
        # columnar rather than isotropic, this ground truth is wrong on that axis.
        return world_pos, v_body_seq, world_to_torus_gt(world_pos, scale)
    
    