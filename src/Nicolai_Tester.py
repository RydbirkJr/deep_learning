import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
import random

# easy to use pool function
def _pool(l_in, scope, kernel_size=(3, 3)):
    return max_pool2d(l_in, kernel_size=kernel_size, scope=scope)  # (3, 3) has shown to work better than (2, 2)

def _conv(l_in, num_outputs, kernel_size, scope, stride=1):
    return convolution2d(l_in, num_outputs=num_outputs, kernel_size=kernel_size,
                         stride=stride, normalizer_fn=batch_norm, scope=scope)

env = gym.make('Pong-v0')
env.reset()
random_episodes = 0
reward_sum = 0
#while random_episodes < 10:
#    env.render()
#    observation, reward, done, _ = env.step(np.random.randint(0,6))
#    reward_sum += reward
#    if done:
#        random_episodes += 1
#        print "Reward for this episode was:",reward_sum
#        reward_sum = 0
#        env.reset()

# hyperparameters
H = 10 # number of hidden layer neurons
batch_size = 10000 # every how many episodes to do a param update?
learning_rate = 1e-2 # feel free to play with this to train faster or more stably.
gamma = 0.99 # discount factor for reward

D = 100800 # input dimensionality

tf.reset_default_graph()

#This defines the network as it goes from taking an observation of the environment to
#giving a probability of chosing to the action of moving left or right.
#observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
#W1 = tf.get_variable("W1", shape=[D, H],
#          initializer=tf.contrib.layers.xavier_initializer())
#layer1 = tf.nn.relu(tf.matmul(observations,W1))
#W2 = tf.get_variable("W2", shape=[H, 1],
#           initializer=tf.contrib.layers.xavier_initializer())
#score = tf.matmul(layer1,W2)
#probability = tf.nn.sigmoid(score)

observations = tf.placeholder(tf.float32, [None, 210, 160, 3], name="x_image_pl")
#is_training_pl = tf.placeholder(tf.bool, name="is_training_pl")

l_conv1_a = _conv(observations, 32, (3, 3), scope="l_conv1_a")
l_pool1 = _pool(l_conv1_a, scope="l_pool1")

l_conv2_a = _conv(l_pool1, 64, (3, 3), scope="l_conv2_a")
print(tf.shape(l_conv2_a))
l_pool2 = _pool(l_conv2_a, scope="l_pool2")

#l_conv3_a = _conv(l_pool2, 128, (3, 3), scope="l_conv3_a")
#l_conv4_a = _conv(l_conv3_a, 128, (3, 3), scope="l_conv4_a")
#l_conv5_a = _conv(l_conv4_a, 128, (3, 3), scope="l_conv5_a")

l_flatten = flatten(l_conv2_a, scope="flatten")

features = batch_norm(l_flatten, scope='features_bn')

l2 = fully_connected(features, num_outputs=128, activation_fn=relu,
                     normalizer_fn=batch_norm, scope="l2")
#l2 = dropout(l2, is_training=is_training_pl, scope="l2_dropout")

# y_ is a placeholder variable taking on the value of the target batch.
probability = fully_connected(l2, 6, activation_fn=softmax, scope="y")

#ConvLayer(3, 3, 6, 32, stride=(1, 1), scope='conv1'),  # out.shape = (B, 210, 160, 32)
#LambdaLayer(tf.nn.sigmoid),
#ConvLayer(2, 2, 32, 64, stride=(2, 2), scope='conv2'),  # out.shape = (B, 105, 80, 64)
#LambdaLayer(tf.nn.sigmoid),
#ConvLayer(3, 3, 64, 64, stride=(1, 1), scope='conv3'),  # out.shape = (B, 105, 80, 64)
#LambdaLayer(tf.nn.sigmoid),
#ConvLayer(2, 2, 64, 128, stride=(2, 2), scope='conv4'),  # out.shape = (B, 53, 40, 128)
#LambdaLayer(tf.nn.sigmoid),
#ConvLayer(3, 3, 128, 128, stride=(1, 1), scope='conv5'),  # out.shape = (B, 53, 40, 128)
#LambdaLayer(tf.nn.sigmoid),
#ConvLayer(2, 2, 128, 256, stride=(2, 2), scope='conv6'),  # out.shape = (B, 27, 20, 256)
#LambdaLayer(tf.nn.sigmoid),
#LambdaLayer(lambda x: tf.reshape(x, [-1, 27 * 20 * 256])),  # out.shape = (B, 27 * 20 * 256)
#Layer(27 * 20 * 256, 6, scope='proj_actions')

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loss = -tf.reduce_mean((tf.log(input_y - probability)) * advantages)
newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 100000
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()  # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        # Make sure the observation is in a shape the network can handle.

        x = np.reshape(observation, [1, 210, 160, 3])

        # Run the policy network and get an action to take.
        tfprob = sess.run(probability, feed_dict={observations: x})

        if tfprob[0][0] > 0.17 or tfprob[0][0] < 0.165:
            print tfprob[0][0]

        action = random.randint(0,5) if max(tfprob[0]) < 0.2 else np.argmax(tfprob[0])# 1 if np.random.uniform() < max(tfprob[0]) else 0
        #action = (np.cumsum(np.asarray(tfprob[0])) > random.randint(0,5)).argmax()

        xs.append(x)  # observation
        y = 1 if action == 0 else 0  # a "fake label"
        ys.append(y)

        # step the environment and get new measurements
        #print action
        observation, reward, done, info = env.step(action)
        if(reward != 0):
            #print action
            print reward
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            print "Done"
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            print "Stacked xs"

            epy = np.vstack(ys)
            print "Stacked ys"

            epr = np.vstack(drs)
            print "Stacked drs"

            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory

            print "Memory reset"
            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            print "Getting gradients"
            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            print "Got gradients"

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                print "Updating policy"
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                print "Grads updated"
                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print 'Average reward for episode %f.  Total average reward %f.' % (
                reward_sum / batch_size, running_reward / batch_size)

                if reward_sum / batch_size > 200:
                    print "Task solved in", episode_number, 'episodes!'
                    break

                reward_sum = 0

            observation = env.reset()

print episode_number, 'Episodes completed.'