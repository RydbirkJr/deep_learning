import gym
from agent import Agent

# init environment
env = gym.make('Pong-v0')

# init agent
agent = Agent(env, batch_size=64, replay_memory_size=312500, scale=1, cropping=(35, 15, 0, 0))
# agent = Agent(env, colors=False, scale=.5, cropping=(30, 10, 6, 6))
# train agent on the environment

#agent.learn(epochs=2, render_test=True, max_test_steps=5000, learning_steps_per_epoch=2000)

agent.learn(epochs=500, render_training=False, render_test=False, learning_steps_per_epoch=10000)
# agent.learn(render_training=True, render_test=True, learning_steps_per_epoch=300)
