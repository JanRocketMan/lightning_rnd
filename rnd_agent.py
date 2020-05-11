import torch
from torch.distributions.categorical import Categorical

from models import CNNPolicyNet, RNDNet


class RNDPPOAgent:
    def __init__(self, action_dim, update_prop, entropy_coeff, ppo_eps):
        self.actor_critic_model = CNNPolicyNet(action_dim)
        self.rnd_model = RNDNet()
        self.device = 'cpu'
        self.mse_crit = torch.nn.MSELoss(reduction='none')
        self.update_proportion = update_prop
        self.ent_coeff = entropy_coeff
        self.ppo_eps = ppo_eps

    def get_action(self, states):
        states = torch.FloatTensor(states).to(self.device)

        policy, value_ext, value_int = self.actor_critic_model(states)
        action_prob = torch.softmax(policy, dim=-1).cpu().numpy()
        action = Categorical(action_prob).sample()

        return [val.cpu().numpy() for val in [action, value_ext, value_int, policy]]

    def get_policy_log_prob(self, actions, policy):
        policy_dist = Categorical(torch.softmax(policy.to(self.device), dim=-1))
        return policy_dist.log_prob(actions).cpu()

    def compute_intrinsic_reward(self, states):
        states = torch.FloatTensor(states).to(self.device)

        int_reward = (
            self.rnd_model.random_net(states) - self.rnd_model.distill_net(states)
        ).pow(2).sum(1) / 2

        return int_reward.cpu().numpy()

    def get_loss(self, data):
        states, actions, ext_target, int_target, total_adv, \
            next_states, log_prob_old = data

        predict_feats, rand_feats = self.rnd_model(next_states)

        curiosity_loss = self.rnd_loss(predict_feats, rand_feats.detach())

        actor_loss, critic_loss, entropy = self.ppo_loss(
            states, actions, ext_target, int_target, total_adv,
            log_prob_old
        )

        return curiosity_loss + actor_loss + critic_loss - self.ent_coeff * entropy

    def rnd_loss(self, predict_feats, rand_feats):
        mse_diff = self.mse_crit(predict_feats, rand_feats.detach()).sum(-1)
        # Drop random observations
        mask = torch.FloatTensor(mse_diff.size(0)).uniform_() > self.update_proportion
        mask = mask.to(self.device)
        return (mse_diff * mask).sum() / torch.max(
            mask.sum(), torch.Tensor([1]).to(self.device)
        )

    def ppo_loss(self, states, actions, ext_target, int_target, total_adv, log_prob_old):
        policy, value_ext, value_int = self.actor_critic_model(states)
        policy_dist = Categorical(torch.softmax(policy, dim=-1))
        entropy = policy_dist.entropy().mean()

        log_prob = policy_dist.log_prob(actions)
        ratio = torch.exp(log_prob - log_prob_old)
        actor_loss = -torch.min(
            ratio * total_adv,
            torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * total_adv
        ).mean()
        critic_ext_loss = self.mse_crit(value_ext.sum(1), ext_target).mean()
        critic_int_loss = self.mse_crit(value_int.sum(1), int_target).mean()
        critic_loss = 0.5 * (critic_ext_loss + critic_int_loss)

        return actor_loss, critic_loss, entropy

    def parameters(self):
        return list(self.rnd_model.parameters()) +\
             list(self.actor_critic_model.parameters())

    def to(self, device):
        self.rnd_model.to(device)
        self.actor_critic_model.to(device)
