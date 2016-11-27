import gym
from random import choice

# init environment
env = gym.make('Pong-v0')

state = env.reset()

terminal = False
while not terminal:
    (_, _, terminal, _) = env.step(choice([1, 2, 3]))
    env.render()
