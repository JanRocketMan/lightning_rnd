import gym
from gym import wrappers
import numpy as np
from PIL import Image
import torch
import cv2

from config import default_config
from rnd_agent import RNDPPOAgent
from environments import MaxAndSkipEnv, MontezumaInfoWrapper


def pre_proc(X, h, w):
    X = np.array(Image.fromarray(X).convert('L')).astype('float32')
    x = cv2.resize(X, (h, w))
    return x / 255


ENV_NAME = default_config["EnvName"]
SAVE_PATH = default_config["SavePath"]
H, W = default_config["ImageHeight"], default_config["ImageWidth"]

env = MontezumaInfoWrapper(MaxAndSkipEnv(gym.make(ENV_NAME), is_render=False), room_address=3)
action_dim = env.action_space.n
device = 'cuda'
agent = RNDPPOAgent(action_dim, device=device)
torch_load = torch.load(SAVE_PATH)
agent.load_state_dict(torch_load["Agent"])
print("N_updates", torch_load["N_Updates"])
agent.actor_critic_model.eval()

env = wrappers.Monitor(env, "./" + ENV_NAME + '_example_run', force=True)
env.reset()
obs = np.zeros((4, H, W), dtype='float32')

total_reward = 0
all_visited_rooms = set()
for i in range(4500):
    with torch.no_grad():
        new_action, _, _, _ = agent.get_action(obs.reshape((1, 4, H, W)))
    action = new_action
    new_obs, reward, done, info = env.step(action)
    total_reward += reward
    obs[:3, :, :] = obs[1:, :, :]
    obs[3, :, :] = pre_proc(new_obs, H, W)
    if done:
        print("Finished, total reward is %d" % total_reward)
        break
    all_visited_rooms.update(info.get('episode', {}).get('visited_rooms', {}))
if not done:
    print("Interrupted after 4500 steps, total reward is %d" % total_reward)
env.close()
print("All visited rooms:", all_visited_rooms)
