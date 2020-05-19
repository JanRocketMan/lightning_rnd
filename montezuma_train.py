import gym
import torch
from torch.multiprocessing import Process, Pipe, set_start_method

from config import default_config
from environments import AtariEnvironmentWrapper
from trainer import RNDTrainer
from rnd_agent import RNDPPOAgent
from env_runner import ParallelEnvironmentRunner, get_default_stored_data
from tensorboardX import SummaryWriter
try:
    set_start_method('spawn')
except RuntimeError:
    pass


NUM_WORKERS = default_config["NumWorkers"]
ENV_NAME = default_config["EnvName"]
EPOCHS = default_config["NumEpochs"]
ROLLOUT_STEPS = default_config["RolloutSteps"]
USETPU = default_config["UseTPU"]
STATE_DICT = default_config.get("StateDict", None)


def train_montezuma():
    env = AtariEnvironmentWrapper(ENV_NAME, False, 0, None)
    action_dim = env.env.action_space.n
    init_state = env.reset()
    env.env.close()
    del env

    if not USETPU:
        opt_device = 'cuda:0'
        run_device = 'cuda:1'
    else:
        opt_device = 'none'
        run_device = 'none'

    print("Initializing agent...")
    agent = RNDPPOAgent(action_dim, device=opt_device)
    frozen_agent = RNDPPOAgent(action_dim, device=run_device)

    writer = SummaryWriter()

    print("Initializing buffer and shared state...")
    with torch.no_grad():
        buffer = get_default_stored_data(NUM_WORKERS, ROLLOUT_STEPS, action_dim)
        for key in buffer.keys():
            buffer[key] = torch.from_numpy(buffer[key]).share_memory_()

    shared_state_dict = {}
    shared_state_dict['agent_state'] = agent.state_dict()
    for key in shared_state_dict['agent_state']['RNDModel'].keys():
        shared_state_dict['agent_state']['RNDModel'][key] = shared_state_dict['agent_state']['RNDModel'][key].share_memory_()
    for key in shared_state_dict['agent_state']['ActorCritic'].keys():
        shared_state_dict['agent_state']['ActorCritic'][key] = shared_state_dict['agent_state']['ActorCritic'][key].share_memory_()

    parent_conn, child_conn = Pipe()

    print("Initializing Environment Runner...")
    env_runner = ParallelEnvironmentRunner(
        NUM_WORKERS, action_dim, ROLLOUT_STEPS, frozen_agent, init_state,
        buffer, shared_state_dict, EPOCHS,
        child_conn, writer,
    )
    print("Done, initializing RNDTrainer...")
    if STATE_DICT is not None:
        state_dict = torch.load(STATE_DICT)
    else:
        state_dict = None

    trainer = RNDTrainer(
        NUM_WORKERS, 4, parent_conn, agent, writer,
        buffer, shared_state_dict, EPOCHS, 
        state_dict=state_dict
    )

    #print("Done, training")

    trainer.start()
    env_runner.run_agent()
    #env_runner.start()

    trainer.join()
    print("Finished!")


if __name__ == '__main__':
    train_montezuma()
