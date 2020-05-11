import numpy as np
import torch
import gym

from config import default_config
from env_runner import ParallelEnvironmentRunner
from utils import RewardForwardFilter, RunningMeanStd, make_train_data

REWARD_DISCOUNT = default_config["RewardDiscount"]
ROLLOUT_STEPS = default_config["RolloutSteps"]
ACTION_DIM = default_config["NumActions"]
EXT_GAMMA = default_config["ExtGamma"]
INT_GAMMA = default_config["IntGamma"]
EXT_COEFF = default_config["ExtCoeff"]
INT_COEFF = default_config["IntCoeff"]

class RNDTrainer:
    def __init__(self, env_runner: ParallelEnvironmentRunner, agent):
        self.env_runner = env_runner
        self.num_workers = self.env_runner.num_workers
        self.agent = agent

        self.current_states = self.env_runner.stored_data["next_states"]
        self.reset_rollout_data()

        self.reward_stats = RunningMeanStd()
        self.disc_reward = RewardForwardFilter(REWARD_DISCOUNT)

        self.n_steps = 0
        self.n_updates = 0
        self.device = 'cpu'

    def reset_rollout_data(self):
        self.stored_data = {
            key: [] for key in [
                'states', 'next_states', 'rewards', 'dones',
                'actions', 'ext_value', 'int_value', 'policy',
                'intrinsic_reward'
            ]
        }

    def accumulate_rollout_data(self):
        self.reset_rollout_data()
        for step in range(ROLLOUT_STEPS + 1):
            # Play next step
            result = self.env_runner.run_agent(
                self.agent, self.current_states, compute_int_reward=True
            )
            # Append new data to existing and update states
            for key in ['ext_value', 'int_value']:
                self.stored_data[key].append(result[key])
            if step < ROLLOUT_STEPS:
                for key in result.keys() - ['ext_value', 'int_value', 'states']:
                    self.stored_data[key].append(result[key])
                self.stored_data['states'].append(self.current_states)
                self.current_states = result['next_states']
            # Log stats
            if False:
                # Todo - add TB logging
                pass

        # Transpose data s.t. first dim corresponds to num_workers
        for key in self.stored_data.keys():
            self.stored_data[key] = np.swapaxes(np.stack(
                self.stored_data[key]), 0, 1
            )
        # Normalize states
        self.stored_data["states"] /= 255.0

    def normalize_intrinsic_rewards(self):
        rewards_per_worker = np.array([
            self.disc_reward.update(reward_per_step) for
            reward_per_step in self.stored_data["intrinsic_reward"].T
        ]).reshape(-1)
        self.reward_stats.update(rewards_per_worker)
        self.stored_data["intrinsic_reward"] /= (self.reward_stats.std + 1e-6)

    def train(self, num_epochs):
        for k in range(num_epochs):
            self.n_steps += (self.env_runner.num_workers * ROLLOUT_STEPS)
            self.n_updates += 1

            with torch.no_grad():
                self.accumulate_rollout_data()

                self.normalize_intrinsic_rewards()

                if False:
                    # Todo - add TB logging
                    pass

                ext_target, ext_adv = make_train_data(
                    self.stored_data["rewards"], self.stored_data["dones"],
                    self.stored_data["ext_value"], EXT_GAMMA,
                    ROLLOUT_STEPS, self.num_workers
                )
                int_target, int_adv = make_train_data(
                    self.stored_data["intrinsic_reward"], np.zeros_like(
                        self.stored_data["intrinsic_reward"]
                    ),
                    self.stored_data["int_value"], INT_GAMMA,
                    ROLLOUT_STEPS, self.num_workers
                )
                total_adv = INT_COEFF * int_adv + EXT_COEFF * ext_adv

            self.train_step(
                ext_target, int_target, total_adv, 
                self.env_runner.preprocess_obs(self.stored_data["next_states"])
            )

    def train_step(self, ext_target, int_target, total_adv, next_states):
        states_tensor = torch.FloatTensor(self.stored_data["states"]).view(
            -1, 4, 84, 84
        ).to(self.device)
        actions_tensor = torch.LongTensor(self.stored_data["actions"]).view(
            -1
        ).to(self.device)
        ext_target, int_target, total_adv = [
            torch.FloatTensor(val).to(self.device) for val in [
                ext_target, int_target, total_adv
            ]
        ]
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        policies_tensor = torch.FloatTensor(self.stored_data["policy"]).view(
            self.stored_data["policy"].shape[0], -1
        ).to(self.device)

        with torch.no_grad():
            log_prob_old = self.agent.get_policy_log_prob(
                policies_tensor
            )

        for i in range(self.epoch_steps):
            # TODO: add ppo + rnd losses comp
            pass

