import gym
from gym import wrappers
import numpy as np
from PIL import Image
import torch

from config import default_config
from rnd_agent import RNDPPOAgent


def pre_proc(X, h, w):
    Img = Image.fromarray(X)
    Img = Img.resize((H, W), Image.LINEAR).convert('L')
    X = np.array(Img).astype('float32')
    return X


ENV_NAME = default_config["EnvName"]
SAVE_PATH = default_config["SavePath"]
H, W = default_config["ImageHeight"], default_config["ImageWidth"]

env = gym.make(ENV_NAME)
action_dim = env.action_space.n
device = 'cuda'
agent = RNDPPOAgent(action_dim, device=device)
agent.load_state_dict(torch.load(SAVE_PATH)["Agent"])

env = wrappers.Monitor(env, "./" + ENV_NAME + '_example_run', force=True)
env.reset()
obs = np.zeros((4, H, W), dtype='float32')

for i in range(4500):
    with torch.no_grad():
        new_action, _, _, _ = agent.get_action(obs.reshape((1, 4, H, W)))
    if np.random.rand() <= 0.25 and i > 0:
        new_action = action
    action = new_action
    new_obs, reward, done, info = env.step(action)
    obs[:3, :, :] = obs[1:, :, :]
    obs[3, :, :] = pre_proc(new_obs, H, W)
    if done: break
env.close()
