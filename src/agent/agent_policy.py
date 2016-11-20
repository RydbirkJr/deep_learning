from agent import Agent
import numpy as np
from PIL import Image
from random import random, randint, choice
import skimage.color
import skimage.transform
from lasagne.layers import get_all_param_values
import pickle
from tqdm import tqdm


class AgentPolicy(Agent):
    def __init__(self, environment, network):
        super(AgentPolicy, self).__init__(environment, network)

    def learn(self,
              epochs=100,
              states_per_batch=10000,
              time_limit=None,
              learning_rate=0.01,
              discount_factor=1.0,
              early_stop=None):
        """
        Learn the given environment by the policy gradient method.
        """
        mean_train_rs = []
        mean_val_rs = []
        self.loss = []

        print "Start training using %d epochs, %d states per batch, %d timelimit, %1.5f learning rate" % (epochs,
            states_per_batch, time_limit, learning_rate)

        best_result = -100  # Just low enough to ensure everything else will be better

        for epoch in xrange(epochs):

            # 1. collect trajectories until we have at least states_per_batch total timesteps
            trajectories = []
            total_trajectories = 0
            total_games = 0
            with tqdm(total=states_per_batch) as pbar:
                while total_trajectories < states_per_batch:
                    trajectory = self.get_trajectory(epoch, epochs, time_limit, deterministic=False)
                    trajectories.append(trajectory)
                    length = len(trajectory["reward"])
                    total_trajectories += length
                    total_games += 1
                    pbar.update(length)
                    # if (total_trajectories / 5000) > steps_since_last:
                    #     steps_since_last = (total_trajectories / 5000)
                    #     print '{0} steps processed'.format(total_trajectories)

            print 'Finished collecting trajectories.'

            all_states = np.concatenate([trajectory["state"] for trajectory in trajectories])

            # 2. compute cumulative discounted rewards (returns)
            rewards = [self._cumulative_discount(trajectory["reward"], discount_factor) for trajectory in trajectories]
            maxlen = max(len(reward) for reward in rewards)
            padded_rewards = [np.concatenate([reward, np.zeros(maxlen - len(reward))]) for reward in rewards]

            # 3. compute time-dependent baseline
            baseline = np.mean(padded_rewards, axis=0)

            # 4. compute advantages
            advs = [reward - baseline[:len(reward)] for reward in rewards]
            all_actions = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_advantages = np.concatenate(advs)

            # 5. do policy gradient update step
            loss = self.network.train(all_states, all_actions, all_advantages, learning_rate)

            train_rs = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # trajectory total rewards
            eplens = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # trajectory lengths

            print("Saving training results...")
            with open("train_results.txt", "w") as train_result_file:
                train_result_file.write(str((train_rs.mean())))

            print "\nTesting..."
            # compute validation reward
            val_reward = np.array(
                [self.get_trajectory(epoch, epochs, time_limit, deterministic=True, render=False)['reward'].sum() for _ in range(1)]
            )

            # update stats
            mean_train_rs.append(train_rs.mean())
            mean_val_rs.append(val_reward.mean())
            self.loss.append(loss)

            if val_reward.max() > best_result:
                print "New best result. Storing weights."
                best_result = val_reward.max()
                pickle.dump(get_all_param_values(self.network.l_out), open('best_weights.dump', "w"))

            print("Saving test results...")
            with open("test_results.txt", "w") as test_result_file:
                test_result_file.write(str((val_reward.mean())))

            print "Saving the network weights..."
            pickle.dump(get_all_param_values(self.network.l_out), open('weights.dump', "w"))

            # print stats
            print '%3d mean_train_r: %6.2f mean_val_r: %6.2f loss: %f games played: %3d' % (
                epoch + 1, train_rs.mean(), val_reward.mean(), loss, total_games)

            # check for early stopping: true if the validation reward has not changed in n_early_stop epochs
            if early_stop and len(mean_val_rs) >= early_stop and \
                    all([x == mean_val_rs[-1] for x in mean_val_rs[-early_stop:-1]]):
                break


    def get_trajectory(self, epoch, epochs, time_limit=None, deterministic=True, render=False):
        """
        Compute state by iteratively evaluating the agent policy on the environment.
        """
        time_limit = time_limit or self.environment.spec.timestep_limit

        # Get stacked initial state
        s1 = self.env_reset()

        trajectory = {'state': [], 'action': [], 'reward': []}

        for _ in xrange(time_limit):
            action = self.get_action(epoch, epochs, s1, deterministic)
            (s2, reward, done, _) = self.environment.step(action + 1)
            if render:
                self.environment.render()

            s2 = self.preprocess(s2)
            s2 = self.add_new_state_to_current(s1, s2)

            trajectory['state'].append(s2)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)

            if done: break

        return {'state': np.array(trajectory['state']),
                'action': np.array(trajectory['action']),
                'reward': np.array(trajectory['reward'])}

    def get_action(self, epoch, epochs, state, deterministic=True):
        """
        Evaluate the agent policy to choose an action, a, given state, s.
        """
        # compute action probabilities
        #action_probabilities = self.network.evaluate(state.reshape(1, -1))

        action_probabilities = self.network.evaluate(np.expand_dims(state, 0))

        if deterministic:
            # choose action with highest probability
            action = action_probabilities.argmax()

        else:
            exp_rate = self.exploration_rate(epoch, epochs)

            if random() <= exp_rate:
                action = randint(0, 2)
            else:
                # Choose the best action according to the network.
                action = action_probabilities.argmax()

            # sample action from cummulative distribution
            #print np.asarray(action_probabilities)
            #action = (np.cumsum(np.asarray(action_probabilities)) > np.random.rand()).argmax()

        return action

    def exploration_rate(self, epoch, epochs):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.01 * epochs  # 10% of learning time
        eps_decay_epochs = 0.9 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    def _cumulative_discount(self, reward, gamma):
        """
        Compute the cumulative discounted rewards.
        """
        reward_out = np.zeros(len(reward), 'float64')
        reward_out[-1] = reward[-1]
        for i in reversed(xrange(len(reward) - 1)):
            reward_out[i] = reward[i] + gamma * reward_out[i + 1]
        return reward_out

    def env_reset(self):
        s1 = self.environment.reset()
        s2, _, _, _ = self.environment.step(choice([1, 2, 3]))
        s3, _, _, _ = self.environment.step(choice([1, 2, 3]))

        res = np.zeros(shape=self.get_state_shape())
        res = res.astype(np.float32)

        res[0] = self.preprocess(s1)
        res[1] = self.preprocess(s2)
        res[2] = self.preprocess(s3)

        return res

    def get_state_shape(self):
        return self.network.shape[1], self.network.shape[2], self.network.shape[3]

    def preprocess(self, img):
        # Crop
        img = img[self.network.cropping[0]:len(img) - self.network.cropping[1], self.network.cropping[2]:len(img[0]) - self.network.cropping[3], 0:]

        # # Scaling
        # if self.scale != 1:
        #     img = skimage.transform.rescale(img, self.scale)

        # This is moved here because of the redef of channels.
        img = skimage.color.rgb2gray(img)

        # This is out because of the redef of channels
        # Grayscale
        # if self.channels == 1:
            # plt.imshow(img)
            # img = skimage.color.rgb2gray(img)
            # plt.imshow(img, cmap=plt.cm.gray)
            # img = img[np.newaxis, ...]
        # else:
        #     img = img.reshape(self.channels, self.resolution[0], self.resolution[1])

        img = img.astype(np.float32)

        return img

    def add_new_state_to_current(self, s1, s2):
        res = np.zeros(shape=self.get_state_shape())
        res = res.astype(np.float32)

        res[0] = s1[1]
        res[1] = s1[2]
        res[2] = s2

        return res

    # def _state_reshape(self, state):
    #     img = Image.fromarray(state, 'RGB').convert('L')
    #     size = (self.network.shape[2], self.network.shape[2])
    #     img.thumbnail(size, Image.ANTIALIAS)
    #     return np.expand_dims(np.array(img), 0)

Agent.register(AgentPolicy)
