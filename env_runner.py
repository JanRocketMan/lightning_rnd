import numpy as np
from torch.multiprocessing import Pipe
from config import default_config

from environments import AtariEnvironmentWrapper
from utils import RunningMeanStd, RewardForwardFilter


ENV_NAME = default_config["EnvName"]
STICKY_ACTION = default_config["UseStickyAction"]
STICKY_ACTION_PROB = default_config["StickyActionProb"]
IMAGE_HEIGHT = default_config["ImageHeight"]
IMAGE_WIDTH = default_config["ImageWidth"]
REWARD_DISCOUNT = default_config["RewardDiscount"]
INIT_STEPS = default_config["NumInitSteps"]
ACTION_DIM = default_config["NumActions"]


class ParallelEnvironmentRunner:
    def __init__(self, num_workers, render_envs=False):
        self.num_workers = num_workers
        self.render_envs = render_envs
        self.all_works = []
        self.parent_conns = []
        self.child_conns = []

        self.reset_stored_data()
        # obs_rms
        self.observation_stats = RunningMeanStd(
            shape=(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        )

        self.__init_workers()
        self.__init_obs_stats()

    def reset_stored_data(self):
        self.stored_data = {
            'next_states': np.zeros(
                (self.num_workers, 4, IMAGE_HEIGHT, IMAGE_WIDTH),
                dtype=np.float32
            ),
            'rewards': np.zeros(self.num_workers, dtype=np.float32),
            'dones': np.zeros(self.num_workers, dtype=np.bool),
            'real_dones': np.zeros(self.num_workers, dtype=np.bool)
        }

    def preprocess_obs(self, some_states):
        return np.clip(
            (some_states[:, 3, :, :] - self.observation_stats.mean) /\
                self.observation_stats.std, -5, 5
        )

    def run_agent(self, agent, states, compute_int_reward=False, update_stats=True):
        actions, ext_value, int_value, policy = agent.get_action(
            states / 255
        )

        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(action)

        for j, parent_conn in enumerate(self.parent_conns):
            new_state, reward, done, real_done = parent_conn.recv()

            self.stored_data['next_states'][j] = new_state
            self.stored_data['rewards'][j] = np.clip(reward, -1, 1)
            self.stored_data['dones'][j] = done
            self.stored_data['real_dones'][j] = real_done

        if update_stats:
            self.observation_stats.update(self.stored_data['next_states'])

        ret_dict = self.stored_data
        ret_dict["actions"] = actions
        ret_dict["ext_value"] = ext_value
        ret_dict["int_value"] = int_value
        ret_dict["policy"] = policy

        if compute_int_reward:
            ret_dict['intrinsic_reward'] = agent.get_intrinsic_reward(
                self.preprocess_obs(self.stored_data['next_states'])
            )

        return ret_dict

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
        class DummyAgent:
            def __init__(self, num_workers):
                self.num_workers = num_workers
            def get_action(self, state):
                return np.random.randint(
                    0, ACTION_DIM, size=(self.num_workers)
                )

        for _ in range(INIT_STEPS):
            _ = self.run_agent(
                DummyAgent(self.num_workers),
                self.stored_data['next_states']
            )
        self.reset_stored_data()
