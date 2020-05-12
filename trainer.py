import numpy as np
import torch
from torch.optim import Adam
from tensorboardX import SummaryWriter

from config import default_config
from env_runner import ParallelEnvironmentRunner
from rnd_agent import RNDPPOAgent
from utils import RewardForwardFilter, RunningMeanStd, make_train_data
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


class RNDTrainer:
    def __init__(self, env_runner: ParallelEnvironmentRunner, agent: RNDPPOAgent,
        logger: SummaryWriter, device: str
    ):
        self.env_runner = env_runner
        self.num_workers = self.env_runner.num_workers
        self.device = device

        self.agent = agent
        self.agent_optimizer = Adam(
            [p for p in self.agent.parameters() if p.requires_grad], lr=LEARNING_RATE
        )

        self.logger = logger
        self.current_states = self.env_runner.stored_data["next_states"]
        self.reset_rollout_data()

        self.reward_stats = RunningMeanStd()
        self.disc_reward = RewardForwardFilter(INT_REWARD_DISCOUNT)

        self.n_steps = 0
        self.n_updates = 0

        self.logged_worker_id = 0
        self.log_reward = 0.0
        self.log_steps = 0
        self.log_episode = 0

    def reset_rollout_data(self):
        self.stored_data = {
            key: [] for key in [
                'states', 'next_states', 'rewards', 'dones', 'real_dones',
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
                self.log_reward += result["rewards"][self.logged_worker_id]
                self.log_steps += 1
                for key in result.keys() - ['ext_value', 'int_value', 'states']:
                    self.stored_data[key].append(result[key])
                self.stored_data['states'].append(self.current_states)
                self.current_states = result['next_states']

            # Log stats
            if result['real_dones'][self.logged_worker_id]:
                self.log_episode += 1
                self.logger.add_scalar(
                    'data/reward_per_episode', self.log_reward, self.log_episode
                )
                self.logger.add_scalar(
                    'data/reward_per_updates', self.log_reward, self.n_updates
                )
                self.logger.add_scalar(
                    'data/num_steps', self.log_steps, self.log_episode
                )
                self.log_reward, self.log_steps = 0.0, 0

        # Transpose data s.t. first dim corresponds to num_workers
        for key in self.stored_data.keys():
            self.stored_data[key] = np.swapaxes(np.stack(
                self.stored_data[key]), 0, 1
            )
        # Normalize states
        self.stored_data["states"] /= 255.0

    def normalize_rewards(self):
        rewards_per_worker = np.array([
            self.disc_reward.update(reward_per_step) for
            reward_per_step in self.stored_data["intrinsic_reward"].T
        ]).reshape(-1)
        self.reward_stats.update(rewards_per_worker)
        self.stored_data["intrinsic_reward"] /= (self.reward_stats.std + 1e-6)
        self.stored_data["rewards"] = np.clip(self.stored_data["rewards"], -1, 1)

    def train(self, num_epochs, state_dict=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)
        start_time = time.time()
        global ROLLOUT_STEPS
        #print("START TRAINING")
        START_ROLLOUT_STEPS = ROLLOUT_STEPS
        for k in range(num_epochs):
            if self.n_updates < 200:
                ROLLOUT_STEPS = int(START_ROLLOUT_STEPS/10)
            else:
                ROLLOUT_STEPS = START_ROLLOUT_STEPS
            self.n_steps += (self.env_runner.num_workers * ROLLOUT_STEPS)
            self.n_updates += 1

            with torch.no_grad():
                #print("ROLLOUT...", time.time() - start_time)
                self.accumulate_rollout_data()
                #print("NORMALIZE...", time.time() - start_time)
                self.normalize_rewards()

                self.logger.add_scalar(
                    'data/intrinsic_reward_per_episode', np.sum(
                        self.stored_data["intrinsic_reward"] / self.num_workers
                    ), self.log_episode
                )
                self.logger.add_scalar(
                    "data/intrinsic_reward_per_rollout", np.sum(
                        self.stored_data["intrinsic_reward"] / self.num_workers
                    ), self.n_updates
                )
                self.logger.add_scalar(
                    "data/max_probability_per_episode",
                    torch.softmax(
                        torch.from_numpy(self.stored_data["policy"]), -1
                    ).max(1)[0].mean().item(), self.log_episode
                )
                #print("MAKE TRAIN...", time.time() - start_time)
                ext_target, ext_adv = make_train_data(
                    self.stored_data["rewards"], self.stored_data["dones"],
                    self.stored_data["ext_value"], EXT_DISCOUNT,
                    ROLLOUT_STEPS, self.num_workers
                )
                int_target, int_adv = make_train_data(
                    self.stored_data["intrinsic_reward"], np.zeros_like(
                        self.stored_data["intrinsic_reward"]
                    ),
                    self.stored_data["int_value"], INT_DISCOUNT,
                    ROLLOUT_STEPS, self.num_workers
                )
                total_adv = INT_COEFF * int_adv + EXT_COEFF * ext_adv
                #print("PACK TO LOADER...", time.time() - start_time)
                c_loader = self.pack_to_dataloader(
                    ext_target, int_target, total_adv, 
                    self.env_runner.preprocess_obs(
                        self.stored_data["next_states"].reshape(-1, 4, 84, 84)
                    )
                )
            #print("MAKE STEP...", time.time() - start_time)
            self.train_step(c_loader)

            if self.n_steps % (self.num_workers * ROLLOUT_STEPS * 100) == 0:
                torch.save(self.state_dict(), SAVE_PATH)

    def pack_to_dataloader(self, ext_target, int_target, total_adv, next_states):
        from torch.utils import data

        states_tensor = torch.FloatTensor(self.stored_data["states"].reshape(-1, 4, 84, 84))
        actions_tensor = torch.LongTensor(self.stored_data["actions"].reshape(-1))
        ext_target, int_target, total_adv = [
            torch.FloatTensor(val) for val in [
                ext_target, int_target, total_adv
            ]
        ]
        next_states_tensor = torch.FloatTensor(next_states)
        policies_tensor = torch.FloatTensor(
            self.stored_data["policy"].reshape(-1, self.stored_data["policy"].shape[-1])
        )

        with torch.no_grad():
            log_prob_old = self.agent.get_policy_log_prob(
                actions_tensor, policies_tensor
            )

        current_data = data.TensorDataset(
            states_tensor, actions_tensor, ext_target, int_target, total_adv,
            next_states_tensor, log_prob_old
        )
        return data.DataLoader(current_data, batch_size=BATCH_SIZE, num_workers=8)

    def train_step(self, dataloader):
        for i in range(EPOCH_STEPS):
            for data in dataloader:
                data = [dt.to(self.device) for dt in data]

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
            "Current_States": self.current_states,
            "Reward_Stats": [self.reward_stats.mean, self.reward_stats.var],
            "EnvObs_Stats": [
                self.env_runner.observation_stats.mean,
                self.env_runner.observation_stats.var
            ],
            "N_Steps": self.n_steps,
            "N_Updates": self.n_updates,
            "Log_Episodes": self.log_episode
        }

    def load_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict["Agent"])
        self.agent_optimizer.load_state_dict(state_dict["Optimizer"])
        self.current_states = state_dict["Current_States"]
        self.reward_stats.mean, self.reward_stats.var = state_dict["Reward_Stats"]
        self.env_runner.observation_stats.mean, \
            self.env_runner.observation_stats.var = state_dict["EnvObs_Stats"]
        self.n_steps = state_dict["N_Steps"]
        self.n_updates = state_dict["N_Updates"]
        self.log_episode = state_dict["Log_Episodes"]
