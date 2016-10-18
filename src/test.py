import gym
import time
import tensorflow as tf

# init and run an example environment
env = gym.make('Pong-v0')
env.reset()
for _ in range(200):
    env.render()
    time.sleep(0.01)
    env.step(env.action_space.sample())
env.render(close=True)