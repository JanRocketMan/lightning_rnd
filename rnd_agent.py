import torch
from torch.distributions.categorical import Categorical

from config import default_config
from models import CNNPolicyNet, RNDNet


UPDATE_PROP = default_config["RNDUpdateProportion"]
ENT_COEFF = default_config["PPOEntropyCoeff"]
PPO_EPS = default_config["PPORewardEps"]

class RNDPPOAgent:
    def __init__(self, action_dim, device):
        self.device = device

        self.actor_critic_model = CNNPolicyNet(action_dim).to(self.device)
        self.rnd_model = RNDNet().to(self.device)

        self.mse_crit = torch.nn.MSELoss(reduction='none')

    def get_action(self, states):
        states = torch.FloatTensor(states).to(self.device)

        policy, value_ext, value_int = self.actor_critic_model(states)
        action = Categorical(torch.softmax(policy, dim=-1)).sample()

        return [val.cpu().numpy() for val in [action, value_ext.squeeze(), value_int.squeeze(), policy]]

    def get_policy_log_prob(self, actions, policy):
        policy_dist = Categorical(torch.softmax(policy.to(self.device), dim=-1))
        return policy_dist.log_prob(actions.to(self.device)).cpu()

    def get_intrinsic_reward(self, states):
        states = torch.FloatTensor(states).to(self.device)

        distill_preds, random_preds = self.rnd_model(states)
        int_reward = (distill_preds - random_preds).pow(2).sum(1) / 2

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

        return curiosity_loss + actor_loss + critic_loss - ENT_COEFF * entropy

    def rnd_loss(self, predict_feats, rand_feats):
        mse_diff = self.mse_crit(predict_feats, rand_feats.detach()).mean(-1)
        # Drop observations randomly
        mask = torch.FloatTensor(mse_diff.size(0)).uniform_() < UPDATE_PROP
        mask = mask.to(self.device)
        return (mse_diff * mask).sum() / max(mask.sum().item(), 1)

    def ppo_loss(self, states, actions, ext_target, int_target, total_adv, log_prob_old):
        policy, value_ext, value_int = self.actor_critic_model(states)
        policy_dist = Categorical(torch.softmax(policy, dim=-1))
        entropy = policy_dist.entropy().mean()

        log_prob = policy_dist.log_prob(actions)
        ratio = torch.exp(log_prob - log_prob_old)
        actor_loss = -torch.min(
            ratio * total_adv,
            torch.clamp(ratio, 1.0 - PPO_EPS, 1.0 + PPO_EPS) * total_adv
        ).mean()
        critic_ext_loss = self.mse_crit(value_ext.sum(1), ext_target).mean()
        critic_int_loss = self.mse_crit(value_int.sum(1), int_target).mean()
        critic_loss = 0.5 * (critic_ext_loss + critic_int_loss)

        return actor_loss, critic_loss, entropy

    def parameters(self):
        return list(self.rnd_model.parameters()) +\
             list(self.actor_critic_model.parameters())

    def state_dict(self):
        return {
            "RNDModel": self.rnd_model.state_dict(),
            "ActorCritic": self.actor_critic_model.state_dict()
        }

    def to(self, device):
        self.rnd_model.to(device)
        self.actor_critic_model.to(device)
        return self

    def load_state_dict(self, state_dict):
        self.rnd_model.load_state_dict(state_dict["RNDModel"])
        self.actor_critic_model.load_state_dict(state_dict["ActorCritic"])
