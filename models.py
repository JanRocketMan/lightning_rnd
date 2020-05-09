import torch.nn as nn


def __ortho_init(model):
        for p in model.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, 0.01)
                nn.init.constant_(p.bias, 0.0)


class CNNPolicyNet(nn.Module):
    def __init__(self, action_dim, hidden_dim=512, nonlinearity=nn.LeakyReLU(0.2)):
        super(CNNPolicyNet, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity

        self.conv_features = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            self.nonlinearity,
            nn.Conv2d(32, 64, 4, 2),
            self.nonlinearity,
            nn.Conv2d(64, 64, 3, 1),
            self.nonlinearity
        )
        self.dense_features = nn.Sequential(
            nn.Linear(7 * 7 * 64, self.hidden_dim // 2),
            self.nonlinearity,
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            self.nonlinearity
        )

        self.actor_model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.nonlinearity,
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        self.critic_model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.nonlinearity,
            nn.Linear(self.hidden_dim, 2)
        )

        __ortho_init(self)

    def forward(self, states):
        z = self.conv_features(states).view(states.size(0), -1)
        z = self.dense_features(z)

        return self.actor_model(z), self.critic_model(z).split(1, dim=1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class RNDNet(nn.Module):
    def __init__(self, hidden_dim=512, nonlinearity=nn.LeakyReLU(0.2)):
        super(RNDNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.random_net, self.distill_net = [nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),
            self.nonlinearity,
            nn.Conv2d(32, 64, 4, 2),
            self.nonlinearity,
            nn.Conv2d(64, 64, 3, 1),
            self.nonlinearity,
            Flatten(),
            nn.Linear(7 * 7 * 64, self.hidden_dim // 2),
            self.nonlinearity,
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            self.nonlinearity
        ) for _ in range(2)]

        __ortho_init(self)

        for param in self.random_net.parameters():
            param.requires_grad_(False)

    def forward(self, next_states):
        return [net(next_states) for net in [self.distill_net, self.random_net]]


if __name__ == '__main__':
    import torch

    ex_cnn = CNNPolicyNet(10)
    ex_rnd = RNDNet()

    ex_inp = torch.randn(128, 4, 84, 84)

    ex_out = ex_cnn(ex_inp)
    ex_out_rnd = ex_rnd(ex_inp.split(1, dim=1)[0].unsqueeze(1))

    print("AC input shape", ex_inp.shape)
    print("AC output shape", ex_out.shape)
    print("AC n params", sum(p.numel() for p in ex_cnn.parameters()))

    print("AC input shape", ex_inp.split(1, dim=1)[0].unsqueeze(1).shape)
    print("AC output shape", ex_out_rnd.shape)
    print("AC n params", sum(p.numel() for p in ex_rnd.parameters()) // 2)
