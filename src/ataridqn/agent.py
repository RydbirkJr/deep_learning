# -*- coding: utf-8 -*-

import pickle
from random import randint, random, choice
from time import time

import numpy as np
import skimage.color
import skimage.transform
import theano
import theano.tensor as T
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from tqdm import trange
from replay_memory import ReplayMemory


class Agent(object):
    """
    Reinforcement Learning Agent

    This agent can learn to solve reinforcement learning tasks from
    OpenAI Gym by applying the policy gradient method.
    """

    def __init__(self, env, scale=1, discount_factor=0.99, learning_rate=0.00025, \
                 replay_memory_size=100000, batch_size=32, cropping=(0, 0, 0, 0), weights_file=None):

        # Create the input variables
        s1 = T.tensor4("States")
        a = T.vector("Actions", dtype="int32")
        q2 = T.vector("Next State's best Q-Value")
        r = T.vector("Rewards")
        isterminal = T.vector("IsTerminal", dtype="int8")

        # Set field values
        # Outcommented because we just wont use colors..
        # if colors:
        #     self.channels = 3
        # else:
        #     self.channels = 1

        self.resolution = ((env.observation_space.shape[0] - cropping[0] - cropping[1]) * scale,
                           (env.observation_space.shape[1] - cropping[2] - cropping[3]) * scale)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.actions = 3 # env.action_space
        self.scale = scale
        self.cropping = cropping
        self.continue_training = False  # Overwritten if weights are given
        self.channels = 1  #Channels because we stack frames

        print("Resolution = " + str(self.resolution))
        print("Channels = " + str(self.channels))

        # Create replay memory which will store the transitions
        self.memory = ReplayMemory(capacity=replay_memory_size, resolution=self.resolution, channels=self.channels)

        # policy network
        l_in = InputLayer(shape=(None, self.channels, self.resolution[0], self.resolution[1]), input_var=s1)
        l_conv1 = Conv2DLayer(l_in, num_filters=64, filter_size=[8, 8], nonlinearity=rectify, stride=4)
        l_conv2 = Conv2DLayer(l_conv1, num_filters=32, filter_size=[4, 4], nonlinearity=rectify, stride=2)
        l_conv3 = Conv2DLayer(l_conv2, num_filters=16, filter_size=[3, 3], nonlinearity=rectify, stride=1)
        l_hid1 = DenseLayer(l_conv3, num_units=256, nonlinearity=rectify)
        self.dqn = DenseLayer(l_hid1, num_units=self.actions, nonlinearity=None)

        if weights_file:
            self.load_weights(weights_file)
            self.continue_training = True

        # Define the loss function
        q = get_output(self.dqn)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q = T.set_subtensor(q[T.arange(q.shape[0]), a], r + discount_factor * (1 - isterminal) * q2)
        loss = squared_error(q, target_q).mean()

        # Update the parameters according to the computed gradient using RMSProp.
        params = get_all_params(self.dqn, trainable=True)
        updates = rmsprop(loss, params, learning_rate)

        # Compile the theano functions
        print "Compiling the network ..."
        self.fn_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
        self.fn_get_q_values = theano.function([s1], q, name="eval_fn")
        self.fn_get_best_action = theano.function([s1], T.argmax(q), name="test_fn")
        print "Network compiled."
        self.env = env

    def load_weights(self, filename):
        set_all_param_values(self.dqn, np.load(str(filename)))

    def get_best_action(self, state):
        return self.fn_get_best_action(state.reshape([1, self.channels, self.resolution[0], self.resolution[1]]))

    '''
        s2 has to have same shape as s1.
    '''
    def learn_from_transition(self, s1, a, s2, s2_isterminal, r):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, s2_isterminal, r)

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)
            q2 = np.max(self.fn_get_q_values(s2), axis=1)
            # the value of q2 is ignored in learn if s2 is terminal
            self.fn_learn(s1, q2, a, r, isterminal)

    def exploration_rate(self, epoch, epochs):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        pct_random_rounds = 0.01

        if self.continue_training:
            start_eps = 0.50
            pct_random_rounds = 0

        const_eps_epochs = pct_random_rounds * epochs  # 10% of learning time
        eps_decay_epochs = 0.9 * epochs  # 90% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    def perform_learning_step(self, epoch, epochs, s1):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        # With probability eps make a random action.
        eps = self.exploration_rate(epoch, epochs)
        if random() <= eps:
            a = randint(0, self.actions - 1)
        else:
            # Choose the best action according to the network.
            a = self.get_best_action(s1)
        (s2, reward, isterminal, _) = self.env.step(a+1)

        #s2 = self.add_new_state_to_current(s1, self.preprocess(s2))
        s2 = self.preprocess(s2)

        s3 = s2 if not isterminal else None
        if isterminal:
            x = 2 # TODO This doesn't do anything?
        self.learn_from_transition(s1, a, s3, isterminal, reward)

        return s2, reward, isterminal

    def preprocess(self, img):

        # Crop
        img = img[self.cropping[0]:len(img) - self.cropping[1], self.cropping[2]:len(img[0]) - self.cropping[3], 0:]

        # Scaling
        if self.scale != 1:
            img = skimage.transform.rescale(img, self.scale)

        # This is moved here because of the redef of channels.
        img = skimage.color.rgb2gray(img)

        # This is out because of the redef of channels
        # Grayscale
        # if self.channels == 1:
            # plt.imshow(img)
            # img = skimage.color.rgb2gray(img)
            # plt.imshow(img, cmap=plt.cm.gray)

        # else:
        #     img = img.reshape(self.channels, self.resolution[0], self.resolution[1])
        img = img[np.newaxis, ...]
        img = img.astype(np.float32)

        return img

    def learn(self, render_training=False, render_test=False, learning_steps_per_epoch=10000, \
              test_episodes_per_epoch=5, epochs=100, max_test_steps=10000):

        print "Starting the training!"

        train_results = []
        test_results = []
        best_result = -100  # Just low enough to ensure everything else will be better

        time_start = time()
        for epoch in range(epochs):
            print "\nEpoch %d\n-------" % (epoch + 1)
            eps = self.exploration_rate(epoch + 1, epochs)
            print "Eps = %.2f" % eps
            train_episodes_finished = 0
            train_scores = []

            print "Training..."
            score = 0
            s1 = self.env.reset()
            s1 = self.preprocess(s1)

            # Because s1 contains the first 3 states
            for learning_step in trange(2, learning_steps_per_epoch):
                s2, reward, isterminal = self.perform_learning_step(epoch, epochs, s1)
                '''
                a = self.get_best_action(s1)
                (s2, reward, isterminal, _) = env.step(a)  # TODO: Check a
                s2 = self.preprocess(s2) if not isterminal else None
                '''
                score += reward
                s1 = s2  # s2 has been shifted into s1, and thus we just replace s1 here

                if (render_training):
                    self.env.render()
                if isterminal:
                    train_scores.append(score)
                    s1 = self.env.reset()
                    s1 = self.preprocess(s1)
                    train_episodes_finished += 1
                    score = 0

            print "%d training episodes played." % train_episodes_finished

            train_scores = np.array(train_scores)

            print "Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max()

            train_results.append((train_scores.mean(), train_scores.std()))

            print("Saving training results...")
            with open("train_results.txt", "w") as train_result_file:
                train_result_file.write(str(train_results))

            test_scores = np.array(self.validate(test_episodes_per_epoch, max_test_steps, render_test))

            if test_scores.max() > best_result:
                print "New best result. Storing weights."
                best_result = test_scores.max()
                pickle.dump(get_all_param_values(self.dqn), open('best_weights.dump', "w"))


            print "Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max()

            test_results.append((test_scores.mean(), test_scores.std()))

            print("Saving test results...")
            with open("test_results.txt", "w") as test_result_file:
                test_result_file.write(str(test_results))

            print "Saving the network weights..."
            pickle.dump(get_all_param_values(self.dqn), open('weights.dump', "w"))

            print "Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0)
            print "Best result so far: mean: %.1f" % (max([item[0] for item in test_results]))

    def env_reset(self):
        s1 = self.env.reset()
        s2, _, _, _ = self.env.step(choice([1, 2, 3]))
        s3, _, _, _ = self.env.step(choice([1, 2, 3]))

        res = np.zeros(shape=(self.channels, self.resolution[0], self.resolution[1]))
        res = res.astype(np.float32)

        res[0] = self.preprocess(s1)
        res[1] = self.preprocess(s2)
        res[2] = self.preprocess(s3)

        return res

    def add_new_state_to_current(self, s1, s2):
        res = np.zeros(shape=(self.channels, self.resolution[0], self.resolution[1]))
        res = res.astype(np.float32)

        res[0] = s1[1]
        res[1] = s1[2]
        res[2] = s2

        return res

    def validate(self, test_episodes_per_epoch=5, max_test_steps=10000, render_test=False):
        print "\nTesting..."
        test_scores = []
        for test_episode in trange(test_episodes_per_epoch):
            s1 = self.env_reset()
            score = 0
            isterminal = False
            frame = 0
            while not isterminal and frame < max_test_steps:
                a = self.get_best_action(s1)
                (s2, reward, isterminal, _) = self.env.step(a+1)

                # I think this covers the statement bellow
                if not isterminal:
                    s2 = self.add_new_state_to_current(s1, self.preprocess(s2))
                else:
                    s2 = None

                # s2 = self.preprocess(s2) if not isterminal else None

                score += reward
                s1 = s2
                if (render_test):
                    self.env.render()
                frame += 1
            test_scores.append(score)
        return test_scores
