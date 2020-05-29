import torch
from copy import deepcopy
import traceback
import threading

from torch.optim import Adam
from tensorboardX import SummaryWriter

from util.config import default_config
from .env_runner import ParallelEnvironmentRunner, get_default_stored_data
from models.rnd_agent import RNDPPOAgent
from util.utils import RewardForwardFilter, RunningMeanStd, make_train_data

from torch import multiprocessing as mp
from torch.multiprocessing import Process


USE_TPU = default_config.get("UseTPU", False)
if USE_TPU:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print_fn = xm.master_print
    save_fn = xm.save
else:
    print_fn = print
    save_fn = torch.save

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
VTRACE = default_config.get("UseVTraceCorrection", False)


def preprocess_obs(some_states, observation_stats):
    obs_mean, obs_std = observation_stats[0], observation_stats[1]
    return torch.clamp(
        (some_states[:, 3, :, :].float().reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH) - obs_mean) /\
            obs_std, -5, 5
    )


def run_rnd_trainer(num_workers, loader_num_workers, conn_to_actor, agent: RNDPPOAgent, buffer, shared_agent: RNDPPOAgent, num_epochs, state_dict=None):
    trainer_cls = RNDTrainer(
        num_workers, loader_num_workers, conn_to_actor, agent, buffer, shared_agent, num_epochs, state_dict
    )
    trainer_cls.train()


class RNDTrainer:
    def __init__(self, num_workers, loader_num_workers, conn_to_actor, agent: RNDPPOAgent,
        buffer, shared_agent: RNDPPOAgent, num_epochs,
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
        self.shared_agent = shared_agent
        self.num_epochs = num_epochs

        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.reward_stats = RunningMeanStd()
        self.disc_reward = RewardForwardFilter(INT_DISCOUNT)

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
            self.shared_agent.load_state_dict(self.agent.state_dict())
            self.conn_to_actor.send(True)

            for k in range(self.num_epochs):
                passed_episodes = self.conn_to_actor.recv() # Wait for actor to infer with current weights

                for key in self.buffer.keys():
                    self.stored_data[key] = self.buffer[key][0] # Retrieve trajectories
                self.shared_agent.load_state_dict(self.agent.state_dict()) # Update actor weights

                if "intrinsic_rewards" in self.stored_data.keys():
                    self.conn_to_actor.send(
                        self.stored_data["intrinsic_rewards"].sum().item() / self.num_workers
                    )
                else:
                    self.conn_to_actor.send(0.0)

                self.n_updates += 1
                self.n_steps += (self.num_workers * ROLLOUT_STEPS)
                self.n_episodes += passed_episodes

                with torch.no_grad():
                    self.get_intrinsic_rewards() # Collect intrinsic rewards with CURRENT Rnd Net
                    self.normalize_rewards()

                    if VTRACE:
                        action, value_ext, value_int, policy = self.agent.get_action(
                            self.stored_data["states"].reshape(-1, 4, 84, 84).float() / 255
                        )
                        action, value_ext, value_int = [
                            torch.from_numpy(val.reshape(self.num_workers, -1)) for val in [action, value_ext, value_int]
                        ]
                        self.stored_data["new_log_prob_policies"] = self.agent.get_policy_log_prob(
                            self.stored_data["actions"].reshape(-1).cpu().numpy(), policy
                        ).reshape(self.num_workers, -1)
                        assert self.stored_data["ext_values"].shape == value_ext.shape
                        assert self.stored_data["int_values"].shape == value_int.shape
                        assert self.stored_data["actions"].shape == action.shape
                        assert self.stored_data["log_prob_policies"].shape == self.stored_data["new_log_prob_policies"].shape
                        self.stored_data["ext_values"] = value_ext
                        self.stored_data["int_values"] = value_int

                    # Estimate advantages recursively (with optional VTrace correction)
                    ext_target, ext_adv = make_train_data(
                        self.stored_data["rewards"], self.stored_data["dones"].float(),
                        self.stored_data["ext_values"], EXT_DISCOUNT,
                        ROLLOUT_STEPS, self.num_workers,
                        log_probs_policies_old=self.stored_data["log_prob_policies"] if VTRACE else None,
                        log_probs_policies=self.stored_data["new_log_prob_policies"] if VTRACE else None
                    )
                    int_target, int_adv = make_train_data(
                        self.stored_data["intrinsic_rewards"], torch.zeros_like(
                            self.stored_data["intrinsic_rewards"]
                        ).float(),
                        self.stored_data["int_values"], INT_DISCOUNT,
                        ROLLOUT_STEPS, self.num_workers,
                        log_probs_policies_old=self.stored_data["log_prob_policies"] if VTRACE else None,
                        log_probs_policies=self.stored_data["new_log_prob_policies"] if VTRACE else None
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

                if self.n_updates % 100 == 0:
                    save_fn(self.state_dict(), SAVE_PATH)

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            print_fn("Exception in actor process")
            traceback.print_exc()
            print_fn()
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
                if USE_TPU:
                    xm.optimizer_step(self.agent_optimizer, barrier=True)
                else:
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
        orig_device = self.agent.device
        self.agent = self.agent.to('cpu')
        self.agent.load_state_dict(state_dict["Agent"])
        self.agent = self.agent.to(orig_device)
        self.agent_optimizer.load_state_dict(state_dict["Optimizer"])
        self.reward_stats.mean, self.reward_stats.var = state_dict["Reward_Stats"]

        self.n_steps = state_dict["N_Steps"]
        self.n_updates = state_dict["N_Updates"]
        self.n_episodes = state_dict["N_Episodes"]
