import typing
import gym
import torch
from torch.multiprocessing import Process, Pipe, set_start_method, get_context
from multiprocessing import Lock

from config import default_config
from environments import AtariEnvironmentWrapper
from trainer import run_rnd_trainer
from rnd_agent import RNDPPOAgent
from env_runner import ParallelEnvironmentRunner, get_default_stored_data
from tensorboardX import SummaryWriter
try:
    set_start_method('spawn')
except RuntimeError:
    pass

USETPU = default_config["UseTPU"]

NUM_WORKERS = default_config["NumWorkers"]
ENV_NAME = default_config["EnvName"]
EPOCHS = default_config["NumEpochs"]
ROLLOUT_STEPS = default_config["RolloutSteps"]
STATE_DICT = default_config.get("StateDict", None)
IMAGE_HEIGHT = default_config["ImageHeight"]
IMAGE_WIDTH = default_config["ImageWidth"]


Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_buffer(W, T, action_dim) -> Buffers:
    specs = dict(
        states=dict(size=(W, T, 4, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.uint8),
        next_states=dict(size=(W, T, 4, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.uint8),
        actions=dict(size=(W, T), dtype=torch.long),
        rewards=dict(size=(W, T), dtype=torch.float32),
        dones=dict(size=(W, T), dtype=torch.bool),
        real_dones=dict(size=(W, T), dtype=torch.bool),
        ext_values=dict(size=(W, T + 1), dtype=torch.float32),
        int_values=dict(size=(W, T + 1), dtype=torch.float32),
        policies=dict(size=(W, T, action_dim), dtype=torch.float32),
        log_prob_policies=dict(size=(W, T), dtype=torch.float32),
        obs_stats=dict(size=(2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32),
    )
    buffers: Buffers = {key: [] for key in specs}
    for key in buffers:
        buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train_montezuma():
    env = AtariEnvironmentWrapper(ENV_NAME, False, 0, None)
    action_dim = env.env.action_space.n
    init_state = torch.from_numpy(env.reset())
    env.env.close()
    del env

    if not USETPU:
        opt_device = 'cuda:0'
        run_device = 'cuda:1'
        print_fn = print
    else:
        import torch_xla
        import torch_xla.core.xla_model as xm
        opt_device = xm.xla_device()
        run_device = xm.xla_device()
        print_fn = xm.master_print

    if STATE_DICT is not None:
        state_dict = torch.load(STATE_DICT, map_location='cpu')
    else:
        state_dict = None

    print_fn("Initializing agent...")
    agent = RNDPPOAgent(action_dim, device=opt_device)

    writer = SummaryWriter()

    print_fn("Initializing buffer and shared state...")
    with torch.no_grad():
        buffer = create_buffer(NUM_WORKERS, ROLLOUT_STEPS, action_dim)

    shared_model = RNDPPOAgent(action_dim, device=run_device)
    shared_model.share_memory()

    parent_conn, child_conn = Pipe()

    print_fn("Initializing Environment Runner...")
    env_runner = ParallelEnvironmentRunner(
        NUM_WORKERS, action_dim, ROLLOUT_STEPS, shared_model, init_state,
        buffer, EPOCHS,
        parent_conn, writer,
    )
    if state_dict and "N_Episodes" in state_dict.keys():
        env_runner.log_episode = state_dict["N_Episodes"]

    print_fn("Done, initializing RNDTrainer...")

    learner = Process(
        target=run_rnd_trainer,
        args=(
            NUM_WORKERS, 4, child_conn, agent,
            buffer, shared_model, EPOCHS, state_dict
        )
    )
    learner.start()

    print_fn("Done, training")

    env_runner.run_agent()

    learner.join()
    print_fn("Finished!")


if __name__ == '__main__':
    train_montezuma()
