import torch
from copy import deepcopy
import traceback
import threading

from torch.optim import Adam
from tensorboardX import SummaryWriter

from config import default_config
from env_runner import ParallelEnvironmentRunner, get_default_stored_data
from rnd_agent import RNDPPOAgent
from utils import RewardForwardFilter, RunningMeanStd, make_train_data

from torch import multiprocessing as mp
from torch.multiprocessing import Process

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
    return torch.clamp(
        (some_states[:, 3, :, :].float().reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH) - obs_mean) /\
            obs_std, -5, 5
    )


def run_rnd_trainer(num_workers, loader_num_workers, conn_to_actor, agent: RNDPPOAgent, buffer, shared_state_dict, num_epochs, state_dict=None):
    trainer_cls = RNDTrainer(
        num_workers, loader_num_workers, conn_to_actor, agent, buffer, shared_state_dict, num_epochs, state_dict
    )
    trainer_cls.train()


class RNDTrainer:
    def __init__(self, num_workers, loader_num_workers, conn_to_actor, agent: RNDPPOAgent,
        buffer, shared_state_dict, num_epochs,
        state_dict=None
    ):
        self.num_workers = num_workers
        self.loader_num_workers = loader_num_workers
        self.conn_to_actor = conn_to_actor

        self.agent = agent
        self.agent_optimizer = Adam(
            [p for p in self.agent.parameters() if p.requires_grad], lr=LEARNING_RATE
        )

        self.n_updates = 0
        self.n_steps = 0
        self.n_episodes = 0
        self.num_epochs = num_epochs

        self.buffer = buffer
        self.shared_state_dict = shared_state_dict
        self.num_epochs = num_epochs

        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.reward_stats = RunningMeanStd()
        self.disc_reward = RewardForwardFilter(INT_REWARD_DISCOUNT)

        self.stored_data = {}

    def get_intrinsic_rewards(self):
        curr_data = self.stored_data["next_states"].reshape(-1, 4, IMAGE_HEIGHT, IMAGE_WIDTH)
        next_rewards = self.agent.get_intrinsic_reward(
            preprocess_obs(curr_data, self.stored_data["obs_stats"])
        )
        self.stored_data["intrinsic_rewards"] = torch.from_numpy(
            next_rewards.reshape(self.stored_data["rewards"].shape)
        )

    def normalize_rewards(self):
        rewards_per_worker = torch.cat([
            self.disc_reward.update(reward_per_step) for
            reward_per_step in self.stored_data["intrinsic_rewards"].T
        ]).reshape(-1)
        self.reward_stats.update(rewards_per_worker)
        self.stored_data["intrinsic_rewards"] /= (self.reward_stats.std + 1e-6)
        self.stored_data["rewards"] = torch.clamp(self.stored_data["rewards"], -1, 1)

    def train(self, lock=threading.Lock()):
        try:
            self.conn_to_actor.send(True)
            print("L -1: Sent to agent that everything is ok")

            for k in range(self.num_epochs):
                passed_episodes = self.conn_to_actor.recv()
                print("L %d: passed episodes" % k, passed_episodes)
                print("L %d: Waited for agent to finish" % k)

                with lock:
                    print("L %d:" % k, [(it[0], it[1].min(), it[1].max()) for it in self.buffer.items()])
                    for key in self.buffer.keys():
                        self.stored_data[key] = deepcopy(self.buffer[key])
                    rnd_shared_state = self.shared_state_dict['agent_state']['RNDModel']
                    agent_shared_state = self.shared_state_dict['agent_state']['ActorCritic']

                    for key in rnd_shared_state.keys():
                        rnd_shared_state[key] = deepcopy(self.agent.state_dict()["RNDModel"][key])
                    for key in agent_shared_state.keys():
                        agent_shared_state[key] = deepcopy(self.agent.state_dict()["ActorCritic"][key])

                print("L %d: states before train step" % k, self.stored_data["states"].min(), self.stored_data["states"].max())
                print("L %d: Loaded stored data & send agent state" % k)

                self.conn_to_actor.send(True)

                self.n_updates += 1
                self.n_steps += (self.num_workers * ROLLOUT_STEPS)
                self.n_episodes += passed_episodes

                with torch.no_grad():
                    self.get_intrinsic_rewards()
                    self.normalize_rewards()
                    ext_target, ext_adv = make_train_data(
                        self.stored_data["rewards"], self.stored_data["dones"].float(),
                        self.stored_data["ext_values"], EXT_DISCOUNT,
                        ROLLOUT_STEPS, self.num_workers
                    )
                    int_target, int_adv = make_train_data(
                        self.stored_data["intrinsic_rewards"], torch.zeros_like(
                            self.stored_data["intrinsic_rewards"]
                        ).float(),
                        self.stored_data["int_values"], INT_DISCOUNT,
                        ROLLOUT_STEPS, self.num_workers
                    )
                    total_adv = INT_COEFF * int_adv + EXT_COEFF * ext_adv
                    c_loader = self.get_dataloader(
                        ext_target, int_target, total_adv, 
                        preprocess_obs(
                            self.stored_data["next_states"].reshape(-1, 4, 84, 84).float(),
                            self.stored_data["obs_stats"]
                        )
                    )
                self.train_step(c_loader)
                print("L %d: Made train step" % k)
                print("L %d: states after tr step" % k, self.stored_data["states"].min(), self.stored_data["states"].max())

                if self.n_updates % 100 == 0:
                    torch.save(self.state_dict(), SAVE_PATH)

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            print("Exception in actor process")
            traceback.print_exc()
            print()
            raise e

    def get_dataloader(self, ext_target, int_target, total_adv, next_states):
        from torch.utils import data

        states_tensor = self.stored_data["states"].reshape(-1, 4, 84, 84).float() / 255
        actions_tensor = self.stored_data["actions"].reshape(-1)
        log_prob_old = self.stored_data["log_prob_policies"].reshape(-1)

        current_data = data.TensorDataset(
            states_tensor, actions_tensor, ext_target, int_target, total_adv,
            next_states, log_prob_old
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
            "Reward_Stats": [self.reward_stats.mean, self.reward_stats.var],
            "EnvObs_Stats": [
                self.stored_data["obs_stats"][0],
                self.stored_data["obs_stats"][1]
            ],
            "N_Steps": self.n_steps,
            "N_Updates": self.n_updates,
            "N_Episodes": self.n_episodes
        }

    def load_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict["Agent"])
        self.agent_optimizer.load_state_dict(state_dict["Optimizer"])
        self.reward_stats.mean, self.reward_stats.var = state_dict["Reward_Stats"]

        self.n_steps = state_dict["N_Steps"]
        self.n_updates = state_dict["N_Updates"]
        self.n_episodes = state_dict["N_Episodes"]
