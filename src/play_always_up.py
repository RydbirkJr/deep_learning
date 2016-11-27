import gym

# init environment
env = gym.make('Pong-v0')


state = env.reset()

terminal = False
while not terminal:
    (_, _, terminal, _) = env.step(2)
    env.render()
