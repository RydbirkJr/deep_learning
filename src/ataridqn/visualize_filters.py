import gym
from agent import Agent
from matplotlib import pyplot as plt
from random import choice
import numpy as np

# init environment
env = gym.make('Pong-v0')

agent = Agent(env, batch_size=32, replay_memory_size=100000, scale=1, cropping=(30, 10, 6, 6), weights_file='best_weightsBreak.dump')

s1 = env.reset()

for _ in range(50):
    s1 = env.step(choice([1,2,3]))

s1 = agent.preprocess(s1[0])

print s1.shape
s1 = np.squeeze(s1, axis=0)
print s1.shape

for layer in agent.conv_layers[2:3]:

    np_W = layer.W.get_value()  # get the filter values from the conv layer
    print np_W.shape, "i.e. the shape is num_filters, num_channels, filter_size, filter_size"
    num_filters, num_channels, filter_size, _ = np_W.shape
    n = int(num_filters ** 0.5)
    print 'n:', n

    # np_W_res = np_W.reshape(n,n,num_channels,filter_size,filter_size)
    fig, ax = plt.subplots(n, n)
    print "learned filter values:"
    for row in range(n):
        for col in range(n):
            n_filter = row * n + col
            #print n_filters
            # if n_filters >= num_filters:
            #     print 'breaking'
            #     break

            ax[row, col].imshow(np_W[n_filter, 0, :, :], cmap='gray', interpolation='none')
            ax[row, col].xaxis.set_major_formatter(plt.NullFormatter())
            ax[row, col].yaxis.set_major_formatter(plt.NullFormatter())


    print 'Moving on..'

    plt.figure()
    plt.imshow(s1, cmap='gray')

    # plt.figure()
    # plt.imshow(s1, cmap='gray', interpolation='none')
    plt.title('Input Image')
    #plt.show()

    # visalize the filters convolved with an input image
    from scipy.signal import convolve2d

    #np_W_res = np_W.reshape(n, n, num_channels, filter_size, filter_size)
    fig, ax = plt.subplots(n, n, figsize=(9, 9))
    print "Response from input image convolved with the filters"
    for row in range(n):
        for col in range(n):
            n_filter = row * n + col
            ax[row, col].imshow(convolve2d(s1, np_W[n_filter, 0,:,:], mode='same'), cmap='gray', interpolation='none')
            ax[row, col].xaxis.set_major_formatter(plt.NullFormatter())
            ax[row, col].yaxis.set_major_formatter(plt.NullFormatter())

    plt.show()