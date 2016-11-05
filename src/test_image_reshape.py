import gym
import numpy as np
from src.agent.agent_policy import AgentPolicy
from src.network.network import Network
from PIL import Image
import timeit


def rgb2gray(rgb):
    #return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b
    #return gray


# init environment
env = gym.make('Pong-v0')

# init agent
# shape = env.observation_space.shape
shape = (None, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])

network = Network(shape, env.action_space.n)
agent = AgentPolicy(env, network)

trajectories = []
total_trajectories = 0
while total_trajectories < 100:
    trajectory = agent.get_trajectory(100, deterministic=True)
    trajectories.append(trajectory)
    total_trajectories += len(trajectory["reward"])

all_states = np.concatenate([trajectory["state"] for trajectory in trajectories])


def convert_states():
    for i in range(100):
        converted = rgb2gray(all_states[i])


def image_convert():
    for i in range(100):
        img = Image.fromarray(all_states[i], 'RGB').convert('L')
        arr = np.array(img)


#img = Image.fromarray(rgb2gray(all_states[20]))
# img.save('before.png')
#img.show()

max = all_states[0].shape[0]

for i in reversed(range(20, 100, 20)):
    size = (i,i)
    img = Image.fromarray(all_states[20], 'RGB').convert('L')
    img.thumbnail(size, Image.ANTIALIAS)
    print 'i: ', i, 'shape: ', np.array(img).shape
    img.show('i: ', i)







