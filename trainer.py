import numpy as np
import torch
from copy import deepcopy
import traceback
from threading import Lock

from torch.optim import Adam
from tensorboardX import SummaryWriter

from config import default_config
from env_runner import ParallelEnvironmentRunner
from rnd_agent import RNDPPOAgent
from utils import RewardForwardFilter, RunningMeanStd, make_train_data

from torch import multiprocessing as mp
from torch.multiprocessing import Process, Pipe, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import time

INT_REWARD_DISCOUNT = default_config["IntrinsicRewardDiscount"]
ROLLOUT_STEPS = default_config["RolloutSteps"]
EXT_DISCOUNT = default_config["ExtRewardDiscount"]
INT_DISCOUNT = default_config["IntRewardDiscount"]
EXT_COEFF = default_config["ExtCoeff"]
INT_COEFF = default_config["IntCoeff"]
LEARNING_RATE = default_config["LearningRate"]
CLIP_GRAD_NORM = default_config["ClipGradNorm"]
BATCH_SIZE = default_config["BatchSize"]
EPOCH_STEPS = default_config["EpochSteps"]
SAVE_PATH = default_config["SavePath"]
IMAGE_HEIGHT = default_config["ImageHeight"]
IMAGE_WIDTH = default_config["ImageWidth"]


def preprocess_obs(some_states, observation_stats):
    obs_mean, obs_std = observation_stats[0], observation_stats[1]
    return np.clip(
        (some_states[:, 3, :, :].reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH) - obs_mean) /\
            obs_std, -5, 5
    )


class RNDTrainer(Process):
    def __init__(self, num_workers, loader_num_workers, conn_to_actor, agent: RNDPPOAgent,
        logger: SummaryWriter, buffer, shared_state_dict, num_epochs, state_dict=None
    ):
        super(RNDTrainer, self).__init__()
        self.daemon = True

        self.num_workers = num_workers
        self.loader_num_workers = loader_num_workers
        self.conn_to_actor = conn_to_actor

        self.agent = deepcopy(agent)
        self.agent_optimizer = Adam(
            [p for p in self.agent.parameters() if p.requires_grad], lr=LEARNING_RATE
        )

        self.logger = logger
        self.n_updates = 0
        self.n_steps = 0

        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.reward_stats = RunningMeanStd()
        self.disc_reward = RewardForwardFilter(INT_REWARD_DISCOUNT)

        self.buffer = buffer
        self.shared_state_dict = shared_state_dict
        self.num_epochs = num_epochs

    def get_intrinsic_rewards(self):
        curr_data = self.stored_data["next_steps"].reshape(-1, 4, IMAGE_HEIGHT, IMAGE_WIDTH)
        next_rewards = self.agent.get_intrinsic_reward(
            preprocess_obs(curr_data.astype('float'), self.stored_data["obs_stats"])
        )
        self.stored_data["intrinsic_rewards"] = next_rewards.reshape(self.stored_data["rewards"].shape)

    def normalize_rewards(self):
        rewards_per_worker = np.array([
            self.disc_reward.update(reward_per_step) for
            reward_per_step in self.stored_data["intrinsic_rewards"].T
        ]).reshape(-1)
        self.reward_stats.update(rewards_per_worker)
        self.stored_data["intrinsic_rewards"] /= (self.reward_stats.std + 1e-6)
        self.stored_data["rewards"] = np.clip(self.stored_data["rewards"], -1, 1)

    def run(self):
        try:
            super(RNDTrainer, self).run()
            self.shared_state_dict['agent_state'] = deepcopy(self.agent.state_dict())
            self.conn_to_actor.send(True)

            for k in range(self.num_epochs):
                finished = self.conn_to_actor.recv()

                self.stored_data = deepcopy(self.buffer.numpy())
                self.shared_state_dict['agent_state'] = deepcopy(self.agent.state_dict())

                self.conn_to_actor.send(True)

                self.n_updates += 1
                self.n_steps += (self.num_workers * ROLLOUT_STEPS)

                with torch.no_grad():
                    self.get_intrinsic_rewards()
                    self.normalize_rewards()
                    ext_target, ext_adv = make_train_data(
                        self.stored_data["rewards"], self.stored_data["dones"],
                        self.stored_data["ext_values"], EXT_DISCOUNT,
                        ROLLOUT_STEPS, self.num_workers
                    )
                    int_target, int_adv = make_train_data(
                        self.stored_data["intrinsic_rewards"], np.zeros_like(
                            self.stored_data["intrinsic_rewards"]
                        ),
                        self.stored_data["int_values"], INT_DISCOUNT,
                        ROLLOUT_STEPS, self.num_workers
                    )
                    total_adv = INT_COEFF * int_adv + EXT_COEFF * ext_adv
                    c_loader = self.pack_to_dataloader(
                        ext_target, int_target, total_adv, 
                        preprocess_obs(
                            self.stored_data["next_states"].reshape(-1, 4, 84, 84).astype('float'),
                            self.stored_data["obs_stats"]
                        )
                    )
                self.train_step(c_loader)

                if self.n_updates % 100 == 0:
                    torch.save(self.state_dict(), SAVE_PATH)
        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            print("Exception in actor process")
            traceback.print_exc()
            print()
            raise e

    def pack_to_dataloader(self, ext_target, int_target, total_adv, next_states):
        from torch.utils import data

        states_tensor = torch.FloatTensor(self.stored_data["states"].reshape(-1, 4, 84, 84).astype('float') / 255)
        actions_tensor = torch.LongTensor(self.stored_data["actions"].reshape(-1))
        ext_target, int_target, total_adv = [
            torch.FloatTensor(val) for val in [
                ext_target, int_target, total_adv
            ]
        ]
        next_states_tensor = torch.FloatTensor(next_states)
        log_prob_old = torch.FloatTensor(
            self.stored_data["log_prob_policies"].reshape(-1)
        )

        current_data = data.TensorDataset(
            states_tensor, actions_tensor, ext_target, int_target, total_adv,
            next_states_tensor, log_prob_old
        )
        return data.DataLoader(current_data, batch_size=BATCH_SIZE, num_workers=self.loader_num_workers)

    def train_step(self, dataloader):
        for i in range(EPOCH_STEPS):
            for data in dataloader:
                data = [dt.to(self.agent.device) for dt in data]

                self.agent_optimizer.zero_grad()
                loss = self.agent.get_loss(data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.agent.parameters() if p.requires_grad],
                    CLIP_GRAD_NORM
                )
                self.agent_optimizer.step()

    def state_dict(self):
        return {
            "Agent": self.agent.state_dict(),
            "Optimizer": self.agent_optimizer.state_dict(),
            "N_Steps": self.n_steps,
            "N_Updates": self.n_updates,
        }

    def load_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict["Agent"])
        self.agent_optimizer.load_state_dict(state_dict["Optimizer"])
        self.n_steps = state_dict["N_Steps"]
        self.n_updates = state_dict["N_Updates"]
