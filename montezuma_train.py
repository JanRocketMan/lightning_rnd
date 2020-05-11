import gym

from config import default_config
from trainer import RNDTrainer
from rnd_agent import RNDPPOAgent
from env_runner import ParallelEnvironmentRunner
from tensorboardX import SummaryWriter


NUM_WORKERS = default_config["NumWorkers"]
ENV_NAME = default_config["EnvName"]
EPOCHS = default_config["NumEpochs"]


def train_montezuma():
    env = gym.make(ENV_NAME)
    action_dim = env.action_space.n
    env.close()

    env_runner = ParallelEnvironmentRunner(NUM_WORKERS, action_dim)
    agent = RNDPPOAgent(action_dim)
    writer = SummaryWriter(logdir='./runs')

    trainer = RNDTrainer(
        env_runner, agent, writer
    )
    trainer.train(EPOCHS)


if __name__ == '__main__':
    train_montezuma()
