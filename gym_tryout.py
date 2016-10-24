import gym

env = gym.make('Pong-v0')
env.reset()
for _ in range(2000):
    env.render()
    _,_,done,_ = env.step(env.action_space.sample())
    if done:
        break
env.render(close=True)