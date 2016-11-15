import gym
from agent import Agent

# init environment
env = gym.make('Pong-v0')

# init agent
agent = Agent(env, batch_size=128, replay_memory_size=312500, colors=False, scale=1, cropping=(30, 10, 6, 6))
# agent = Agent(env, colors=False, scale=.5, cropping=(30, 10, 6, 6))
# train agent on the environment
agent.learn(epochs=500, render_training=False, render_test=False)
# agent.learn(render_training=True, render_test=True, learning_steps_per_epoch=300)
