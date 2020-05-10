import numpy as np
import torch
import gym

from .agents import RNDAgent
from .config import default_config
from .env_runner import ParallelEnvironmentRunner
from .utils import RewardForwardFilter, RunningMeanStd, make_train_data


ROLLOUT_STEPS = default_config["RolloutSteps"]
ACTION_DIM = default_config["NumActions"]


class RNDTrainer:
    def __init__(self, env_runner: ParallelEnvironmentRunner):
        self.n_steps = 0
        self.n_updates = 0
        self.env_runner = env_runner
        env = gym.make(default_config['EnvName'])

        self.input_size = env.observation_space.shape  
        self.output_size = env.action_space.n 
        env.close()
        
        self.agent = RNDAgent(
            self.input_size,
            self.output_size,
            num_env=self.env_runner.num_workers,
            num_step=ROLLOUT_STEPS,
            gamma=float(default_config['gamma']),
            lam=default_config['lam'],
            learning_rate=default_config['learning_rate'],
            ent_coef=default_config['entropy_coef'],
            clip_grad_norm=default_config['clip_grad_norm'],
            epoch=default_config['epoch'],
            batch_size=default_config['batch_size'],
            ppo_eps=default_config['ppo_eps'],
            use_cuda=default_config['use_cuda'],
            use_gae=default_config['use_gae'],
            use_noisy_net=default_config['use_noisy_net']
        ) 

        self.ext_coef = float(default_config['ExtCoef'])
        self.int_coef = float(default_config['IntCoef'])
        self.gamma = float(default_config['gamma'])
        self.int_gamma = float(default_config['int_gamma'])
        self.current_states = self.env_runner.stored_data["next_states"]
        self.stored_rollout_data["states"] = self.stored_rollout_data["next_states"]

    def train(self, num_epochs):
        sample_episode = 0
        sample_rall = 0
        sample_step = 0
        sample_env_idx = 0
        sample_i_rall = 0
        discounted_reward = RewardForwardFilter(default_config["RewardDiscount"])
        reward_stats = RunningMeanStd()

        for k in range(num_epochs):
            total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
                [], [], [], [], [], [], [], [], [], [], []
            self.n_steps += (self.env_runner.num_workers * ROLLOUT_STEPS)
            self.n_updates += 1

            """Accumulate statistics for training"""
            with torch.no_grad():
                for step in range(ROLLOUT_STEPS + 1):
                    #self.stored_rollout_data["states"][step] = self.current_states
                    # dict with 
                    # actions, ext_value, int_value, policy
                    # next_states, rewards, dones, real_dones
                    # (agent, states, compute_int_reward=False, update_stats=True)
                    result = self.env_runner.run_agent(
                        self, self.current_states, compute_int_reward=True
                    )
                    # APPEND NEW DATA
                    #for key in result.keys():
                    #    self.stored_rollout_data[key][step] = result
                    next_states = np.stack(result['next_states'])
                    real_dones = np.hstack(result['real_dones'])
                    total_next_obs.append(np.stack(result['next_obs']))
                    total_int_reward.append(np.hstack(result['intrinsic_reward']))
                    total_state.append(self.current_states)
                    total_reward.append(np.hstack(result['rewards']))
                    total_done.append(np.hstack(result['dones']))
                    total_action.append(result['actions'])
                    total_ext_values.append(result['ext_value'])
                    total_int_values.append(result['int_value'])
                    total_policy.append(result['policy'])
                    total_policy_np.append(result['policy'].cpu().numpy())


                    if step < ROLLOUT_STEPS:
                        self.current_states = next_states[:, :, :, :]
                    if True:
                        # Todo - add TB logging
                        pass
            
                """Transpose and preprocess intrinsic reward, make dataset"""
            # calculate last next value
            _, ext_value, int_value, _ = agent.get_action(np.float32(self.current_states) / 255.)
            total_ext_values.append(ext_value)
            total_int_values.append(int_value)

            total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_reward = np.stack(total_reward).transpose().clip(-1, 1)
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_done = np.stack(total_done).transpose()
            total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
            total_ext_values = np.stack(total_ext_values).transpose()
            total_int_values = np.stack(total_int_values).transpose()
            total_logging_policy = np.vstack(total_policy_np)

            # Step 2. calculate intrinsic reward
            # running mean intrinsic reward
            total_int_reward = np.stack(total_int_reward).transpose()
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                            total_int_reward.T])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_stats.update_from_moments(mean, std ** 2, count)

            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_stats.var)

            # Step 3. make target and advantage
            # extrinsic reward calculate
            ext_target, ext_advantage = make_train_data(total_reward,
                                                total_done,
                                                total_ext_values,
                                                gamma,
                                                ROLLOUT_STEPS,
                                                self.env_runner.num_workers)
            # intrinsic reward calculate
            # None Episodic
            int_target, int_advantage = make_train_data(total_int_reward,
                                                np.zeros_like(total_int_reward),
                                                total_int_values,
                                                int_gamma,
                                                ROLLOUT_STEPS,
                                                self.env_runner.num_workers)

            total_adv = int_advantage * self.int_coef + ext_advantage * self.ext_coef

            # Step 4. update obs normalize param
            self.env_runner.observation_stats.update(total_next_obs)

            # Step 5. Training!
            agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                            total_adv, self.env_runner.preprocess_obs((total_next_obs)),
                            total_policy)
