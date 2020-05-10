from config import default_config
import numpy as np

import torch
from torch._six import inf

from config import default_config


LAMBDA = float(default_config['PPOAdvLambda'])


def make_train_data(reward, done, value, gamma, num_step, num_worker):
    discounted_return = np.empty([num_worker, num_step])
    # PPO Discounted Return 
    generalized_advanced_estimation = np.zeros_like([num_worker, ])
    for t in range(num_step - 1, -1, -1):
        delta = -value[:, t] + reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t])
        generalized_advanced_estimation = delta + gamma * LAMBDA * (1 - done[:, t]) * generalized_advanced_estimation
        discounted_return[:, t] = generalized_advanced_estimation + value[:, t]
    advantage = discounted_return - value[:, :-1]
    return discounted_return.reshape([-1]), advantage.reshape([-1])


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype='float64')
        self.var = np.ones(shape, dtype='float64')
        self.count = epsilon

    def update(self, x):
        batch_mean, batch_var = np.mean(x, axis=0), np.var(x, axis=0)
        self.update_from_moments(batch_mean, batch_var, x.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        M2 = self.var * (self.count) + batch_var * (batch_count) + np.square(delta) * self.count * batch_count / (self.count + batch_count)
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


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm
