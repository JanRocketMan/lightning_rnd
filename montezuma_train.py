import gym

from config import default_config
from trainer import RNDTrainer
from rnd_agent import RNDPPOAgent
from env_runner import ParallelEnvironmentRunner
from tensorboardX import SummaryWriter


NUM_WORKERS = default_config["NumWorkers"]
ENV_NAME = default_config["EnvName"]
EPOCHS = default_config["NumEpochs"]
USETPU = default_config["UseTPU"]
STATE_DICT = default_config.get("StateDict", None)


def train_montezuma():
    env = gym.make(ENV_NAME)
    action_dim = env.action_space.n
    env.close()

    if not USETPU:
        device = 'cuda'
    else:
        device = 'none'

    print("Initializing Environment Runner...")
    env_runner = ParallelEnvironmentRunner(NUM_WORKERS, action_dim)
    print("Done, initializing RNDTrainer...")
    agent = RNDPPOAgent(action_dim, device=device)
    writer = SummaryWriter()

    trainer = RNDTrainer(
        env_runner, agent, writer, device
    )
    print("Done, training")
    trainer.train(EPOCHS, state_dict=STATE_DICT)
    print("Finished!")


if __name__ == '__main__':
    train_montezuma()
