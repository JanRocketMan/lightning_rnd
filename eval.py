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
USETPU = default_config["UseTPU"]
H, W = default_config["ImageHeight"], default_config["ImageWidth"]

env = MontezumaInfoWrapper(MaxAndSkipEnv(gym.make(ENV_NAME), is_render=False), room_address=3)
action_dim = env.action_space.n
agent = RNDPPOAgent(action_dim, device='cpu')
agent_state = torch.load(SAVE_PATH, map_location='cpu')

if "module." in list(agent_state["Agent"]["RNDModel"].keys())[0]:
    # Fix loading if we store dataparallel model
    for key, item in agent_state["Agent"].items():
        new_item = {}
        for key_1, item_1 in item.items():
            new_item[key_1.replace("module.", "")] = item_1
        agent_state["Agent"][key] = new_item
agent.load_state_dict(agent_state["Agent"])

if not USETPU:
    device = 'cuda'
    print_fn = print
else:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print_fn = xm.master_print

agent = agent.to(device)

print_fn("N_updates " + str(agent_state["N_Updates"]))
agent.actor_critic_model.eval()

env = wrappers.Monitor(env, "./" + ENV_NAME + '_example_run', force=True)
env.reset()
obs = np.zeros((4, H, W), dtype='float32')

total_reward = 0
all_visited_rooms = set()
for i in range(6000):
    with torch.no_grad():
        new_action, _, _, _ = agent.get_action(obs.reshape((1, 4, H, W)))
    action = new_action
    new_obs, reward, done, info = env.step(action)
    total_reward += reward
    obs[:3, :, :] = obs[1:, :, :]
    obs[3, :, :] = pre_proc(new_obs, H, W)
    if done:
        print_fn("Finished, total reward is %d" % total_reward)
        break
    all_visited_rooms.update(info.get('episode', {}).get('visited_rooms', {}))
if not done:
    print_fn("Interrupted after 6000 steps, total reward is %d" % total_reward)
env.close()
print_fn("All visited rooms: " + str(all_visited_rooms))
