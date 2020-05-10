import numpy as np
import torch

from config import default_config
from env_runner import ParallelEnvironmentRunner


ROLLOUT_STEPS = default_config["RolloutSteps"]
ACTION_DIM = default_config["NumActions"]


class RNDTrainer:
    def __init__(self, env_runner: ParallelEnvironmentRunner):
        self.n_steps = 0
        self.n_updates = 0
        self.env_runner = env_runner

        self.stored_rollout_data = {
            key: np.concatenate([val[np.newaxis, :] for _ in range(ROLLOUT_STEPS)])
            for key, val in self.env_runner.stored_data.items()
        }
        self.stored_rollout_data["actions"] = np.zeros(
            (ROLLOUT_STEPS, self.env_runner.num_workers)
        )
        self.stored_rollout_data["ext_value"], self.stored_rollout_data["int_value"] = [
            np.zeros(
                (ROLLOUT_STEPS, self.env_runner.num_workers),
                dtype=np.float32
            )
            for _ in range(2)
        ]
        self.stored_rollout_data["policy"] = np.zeros(
            (ROLLOUT_STEPS, self.env_runner.num_workers, ACTION_DIM),
            dtype=np.float32
        )

        self.current_states = self.env_runner.stored_data["next_states"]
        self.stored_rollout_data["states"] = self.stored_rollout_data["next_states"]

    def train(self, num_epochs):

        for k in range(num_epochs):
            self.n_steps += (self.env_runner.num_workers * ROLLOUT_STEPS)
            self.n_updates += 1

            """Accumulate statistics for training"""
            with torch.no_grad():
                for step in range(ROLLOUT_STEPS + 1):
                    self.stored_rollout_data["states"][step] = self.current_states

                    result = self.env_runner.run_agent(
                        self, self.current_states, compute_int_reward=True
                    )
                    # APPEND NEW DATA
                    for key in result.keys():
                        self.stored_rollout_data[key][step] = result

                    if step < ROLLOUT_STEPS:
                        self.current_states = result["next_states"]
                    if True:
                        # Todo - add TB logging
                        pass
            
                """Transpose and preprocess intrinsic reward, make dataset"""
        
            """Do actual training"""

