from network.policy_network import PolicyGradient
import tensorflow as tf
import numpy as np
import gym
from collections import deque
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax

env_name = 'CartPole-v0'
env = gym.make(env_name)

sess = tf.Session()
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
writer = tf.train.SummaryWriter("/tmp/{}-experiment-1".format(env_name))

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n


def hidden_layer(l_in, number_in, number_output, non_lin, number_layer):
    W = tf.get_variable("W" + str(number_layer), [number_in, number_output],
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable("b" + str(number_layer), [number_output],
                        initializer=tf.constant_initializer(0))
    h = non_lin(tf.matmul(l_in, W) + b)
    return h


def policy_network(states):
    h1 = hidden_layer(l_in=states, number_in=state_dim, number_layer=1, number_output=20, non_lin=relu)
    h2 = hidden_layer(l_in=h1, number_in=20, number_layer=2, number_output=num_actions, non_lin=relu)

    # define policy neural network
    # Redo this in a nicer way with nice implementations of layers in TF
    # W1 = tf.get_variable("W1", [state_dim, 20],
    #                      initializer=tf.random_normal_initializer())
    # b1 = tf.get_variable("b1", [20],
    #                      initializer=tf.constant_initializer(0))
    # h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
    # W2 = tf.get_variable("W2", [20, num_actions],
    #                      initializer=tf.random_normal_initializer(stddev=0.1))
    # b2 = tf.get_variable("b2", [num_actions],
    #                      initializer=tf.constant_initializer(0))
    # p = tf.matmul(h1, W2) + b2
    # return p
    return h2

pg = PolicyGradient(sess, optimizer, policy_network, state_dim, num_actions, summary_writer=writer)

MAX_EPOCS = 1000
MAX_STEPS = 400

episode_history = deque(maxlen=100)
for i_episode in xrange(MAX_EPOCS):
    # initialize
    state = env.reset()
    total_rewards = 0

    for t in xrange(MAX_STEPS):
        env.render()
        action = pg.sample_action(state[np.newaxis,:])
        next_state, reward, done, _ = env.step(action)

        total_rewards += reward
        reward = -10 if done else 0.1 # normalize reward
        pg.store_rollout(state, action, reward)

        state = next_state
        if done:
            break

    pg.update_model()
    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)

    print("Episode {}".format(i_episode))
    print("Finished after {} timesteps".format(t+1))
    print("Reward for this episode: {}".format(total_rewards))
    print("Average reward for last 100 episodes: {}".format(mean_rewards))
    if mean_rewards >= 195.0 and len(episode_history) >= 100:
        print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
        break

