import traceback
import threading
from copy import deepcopy
import numpy as np

from torch.multiprocessing import Pipe
import torch
from config import default_config

from rnd_agent import RNDPPOAgent
from environments import AtariEnvironmentWrapper
from utils import RunningMeanStd


ENV_NAME = default_config["EnvName"]
STICKY_ACTION = default_config["UseStickyAction"]
STICKY_ACTION_PROB = default_config["StickyActionProb"]
IMAGE_HEIGHT = default_config["ImageHeight"]
IMAGE_WIDTH = default_config["ImageWidth"]
INIT_STEPS = default_config["NumInitSteps"]


def get_default_stored_data(W, T, action_dim):
    return {
        'states': torch.zeros(
            (W, T, 4, IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=torch.uint8
        ),
        'next_states': torch.zeros(
            (W, T, 4, IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=torch.uint8
        ),
        'actions': torch.zeros((W, T), dtype=torch.long),
        'rewards': torch.zeros((W, T)).float(),
        'dones': torch.zeros((W, T), dtype=torch.bool),
        'real_dones': torch.zeros((W, T), dtype=torch.bool),
        'ext_values': torch.zeros((W, T + 1)).float(),
        'int_values': torch.zeros((W, T + 1)).float(),
        'policies': torch.zeros((W, T, action_dim)).float(),
        'log_prob_policies': torch.zeros((W, T)).float(),
        'obs_stats': torch.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH)).float()
    }


class ParallelEnvironmentRunner:
    def __init__(
        self, num_workers: int, action_dim: int, rollout_steps: int,
        shared_agent: RNDPPOAgent, init_state: torch.ByteTensor,
        buffer, num_epochs,
        conn_to_learner, writer, render_envs=False,
    ):
        self.num_workers = num_workers
        self.render_envs = render_envs
        self.action_dim = action_dim
        self.rollout_steps = rollout_steps
        self.writer = writer
        self.conn_to_learner = conn_to_learner
        self.actor_agent = shared_agent
        self.all_works = []
        self.parent_conns = []
        self.child_conns = []
 
        # Normalization statistics
        self.observation_stats = RunningMeanStd(
            shape=(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        )
        self.init_state = init_state
        self.reset_current_state()

        self.log_env = 0
        self.log_episode, self.log_steps = 0, 0
        self.passed_episodes = 0
        self.log_reward, self.log_total_steps = 0.0, 0

        self.__init_workers()
        self.__init_obs_stats()

        self.buffer = buffer
        self.num_epochs = num_epochs
        with threading.Lock():
            self.actor_agent.reset_params()
            print("Init state dict:", self.actor_agent.state_dict()["RNDModel"]['distill_net.9.weight'][:6, 0])

    def reset_current_state(self):
        self.current_state = torch.zeros(
            (self.num_workers,) + self.init_state.shape,
            dtype=torch.uint8
        )
        for w in range(self.num_workers):
            self.current_state[w] = self.init_state

    def reset_stored_data(self):
        self.stored_data = get_default_stored_data(self.num_workers, self.rollout_steps, self.action_dim)

    def push_to_stored_data(self, key, data, step_idx, worker_idx=None):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if worker_idx is not None:
            self.stored_data[key][worker_idx, step_idx] = data
        else:
            self.stored_data[key][:, step_idx] = data

    def collect_env_results(self, actions, step_idx):
        """Synchronizes environments after each step"""
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(action)

        for j, parent_conn in enumerate(self.parent_conns):
            new_state, reward, done, real_done = parent_conn.recv()

            self.push_to_stored_data('next_states', new_state, step_idx, j)
            self.push_to_stored_data('rewards', reward, step_idx, j)
            self.push_to_stored_data('dones', done, step_idx, j)
            self.push_to_stored_data('real_dones', real_done, step_idx, j)

    def collect_agent_results(self, step_idx, action, state, ext_value, int_value, policy, log_prob_policy):
        # Collect now model-based data
        self.push_to_stored_data('actions', action, step_idx)
        self.push_to_stored_data('states', state, step_idx)
        self.push_to_stored_data('ext_values', ext_value, step_idx)
        self.push_to_stored_data('int_values', int_value, step_idx)
        self.push_to_stored_data('policies', policy, step_idx)
        self.push_to_stored_data('log_prob_policies', log_prob_policy, step_idx)

    def run_agent_step(self, step_idx, action_fn, compute_int_reward=False, compute_agent_outputs=True, update_stats=True):
        # Predict next actions via current agent
        with torch.no_grad():
            state = self.current_state.numpy()
            action, ext_value, int_value, policy = action_fn(
                state.astype('float') / 255
            )

        # Run and collect results across environments
        self.collect_env_results(action, step_idx)
        # Collect now model-based data
        if compute_agent_outputs:
            with torch.no_grad():
                log_prob_policy = self.actor_agent.get_policy_log_prob(
                    action, policy
                )
            self.collect_agent_results(step_idx, action, state, ext_value, int_value, policy, log_prob_policy)

        # Update observation stats
        if update_stats:
            self.observation_stats.update(
                self.stored_data['next_states'][:, step_idx, 3, :, :].reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH).float()
            )
            self.stored_data['obs_stats'][0] = self.observation_stats.mean.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            self.stored_data['obs_stats'][1] = self.observation_stats.std.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

        if compute_int_reward:
            int_rew = self.actor_agent.get_intrinsic_reward(
                self.preprocess_obs(self.stored_data['next_states'][:, step_idx])
            )
            self.push_to_stored_data('int_rewards', int_rew, step_idx)

    def run_agent(self, lock=threading.Lock()):
        try:
            for i in range(self.num_epochs):
                finished = self.conn_to_learner.recv()

                #print("A %d: Waited for learner to finish" % i)

                self.log_total_steps += 1
                self.reset_stored_data()

                #print("A %d: Loaded new actor state" % i)

                for idx in range(self.rollout_steps):
                    self.run_agent_step(
                        idx, self.actor_agent.get_action
                    )
                    self.current_state = self.stored_data["next_states"][:, idx]

                    self.log_current_results(idx)

                with torch.no_grad():
                    _, ext_value, int_value, _ = self.actor_agent.get_action(
                        self.current_state.numpy().astype('float') / 255
                    )
                    self.push_to_stored_data('ext_values', ext_value, self.rollout_steps)
                    self.push_to_stored_data('int_values', int_value, self.rollout_steps)

                #print("A %d: states" % i, self.stored_data["states"].min(), self.stored_data["states"].max())

                for key in self.stored_data.keys():
                    self.buffer[key][0][...] = self.stored_data[key]
                #print("A %d:" % i, [(it[0], it[1][0].min(), it[1][0].max()) for it in self.buffer.items()])
                #print("A %d: agent state" % i, self.actor_agent.state_dict()["RNDModel"]['distill_net.9.weight'][:6, 0])

                #print("A %d: Updated trajectories" % i)

                self.conn_to_learner.send(
                    self.passed_episodes
                )
                self.passed_episodes = 0
                #print("A %d: End of step" % i)

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            print("Exception in actor process")
            traceback.print_exc()
            print()
            raise e

    def log_current_results(self, step_idx):
        self.log_reward += self.stored_data['rewards'][self.log_env, step_idx].item()
        self.log_steps += 1
        if self.stored_data['real_dones'][self.log_env, step_idx]:
            self.log_episode += 1
            self.passed_episodes += 1
            self.writer.add_scalar('data/reward_per_rollout', self.log_reward, self.log_total_steps)
            self.writer.add_scalar('data/reward_per_episode', self.log_reward, self.log_episode)
            self.writer.add_scalar('data/steps_per_episode', self.log_steps, self.log_episode)
            self.writer.add_scalar(
                'data/max_prob_per_episode', torch.softmax(
                    self.stored_data["policies"][:, step_idx],
                    -1
                ).max(1)[0].mean().item()
            )
            self.log_reward, self.log_steps = 0.0, 0

    def __init_workers(self):
        for i in range(self.num_workers):
            parent_conn, child_conn = Pipe()
            work = AtariEnvironmentWrapper(
                ENV_NAME, self.render_envs, i,
                child_conn, sticky_action=STICKY_ACTION,
                p=STICKY_ACTION_PROB
            )
            work.start()
            self.all_works.append(work)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

    def __init_obs_stats(self):
        rand_act = lambda x: (torch.randint(
            0, self.action_dim, (self.num_workers,)
        ), 0.0, 0.0, 0.0)

        self.reset_stored_data()
        for _ in range(INIT_STEPS):
            self.run_agent_step(
                0,
                rand_act,
                compute_agent_outputs=False
            )
            self.current_state = self.stored_data["next_states"][:, 0]
        self.reset_stored_data()
        self.reset_current_state()

    def preprocess_obs(self, some_states):
        return torch.clamp(
            (some_states[:, 3, :, :].reshape(-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH).float() - self.observation_stats.mean) /\
                self.observation_stats.std, -5, 5
        )

    def join_all_workers(self):
        for work in self.all_works:
            work.join(timeout=1)

    def __del__(self):
        self.join_all_workers()
        self.conn_to_learner.close()
