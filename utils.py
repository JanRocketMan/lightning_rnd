from config import default_config

import torch
from torch._six import inf

from config import default_config


LAMBDA = float(default_config['PPOAdvLambda'])
VTRACE = default_config.get("UseVTraceCorrection", False)


def make_train_data(
    reward, done, value, discount,
    num_steps, num_workers, 
    log_probs_policies_old=None, log_probs_policies=None
):
    discounted_return = torch.empty([num_workers, num_steps])
    # PPO Discounted Return 
    generalized_advanced_estimation = torch.zeros((num_workers)).float()

    if VTRACE:
        delta_coeffs = torch.clamp_min(
            torch.exp(log_probs_policies - log_probs_policies_old),
            1.0
        )

    for t in range(num_steps - 1, -1, -1):
        delta = -value[:, t] + reward[:, t] + discount * value[:, t + 1] * (1 - done[:, t])
        generalized_advanced_estimation = delta + discount * LAMBDA * (1 - done[:, t]) * generalized_advanced_estimation
        if VTRACE:
            generalized_advanced_estimation *= delta_coeffs[:, t]
        discounted_return[:, t] = generalized_advanced_estimation + value[:, t]
    advantage = discounted_return - value[:, :-1]
    return discounted_return.reshape([-1]), advantage.reshape([-1])


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float()
        self.var = torch.ones(shape).float()
        self.count = epsilon

    @property
    def std(self):
        return self.var ** 0.5

    def update(self, x):
        batch_mean, batch_var = torch.mean(x, dim=0), torch.var(x, dim=0)
        self.update_from_moments(batch_mean, batch_var, x.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        M2 = self.var * (self.count) + batch_var * (batch_count) + delta.pow(2) * self.count * batch_count / (self.count + batch_count)
        self.var = M2 / (self.count + batch_count)
        self.count = batch_count + self.count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
