from agent import Agent
import numpy as np
from PIL import Image
import random
from random import sample, randint, random

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
        self.epochs = epochs
        self.epoch = 1
        print "Start training using %d epochs, %d states per batch, %d timelimit, %1.5f learning rate" % (epochs,
            states_per_batch, time_limit, learning_rate)

        for epoch in xrange(epochs):
            self.epoch = epoch
            # 1. collect trajectories until we have at least states_per_batch total timesteps
            trajectories = []
            total_trajectories = 0
            total_games = 0
            while total_trajectories < states_per_batch:
                print total_trajectories,"/",states_per_batch
                trajectory = self.get_trajectory(time_limit, deterministic=False)
                trajectories.append(trajectory)
                total_trajectories += len(trajectory["reward"])
                total_games += 1

            ####TEST####

            '''size = 1000
            if(len(trajectories)<1000):
                size = len(trajectories)
            #print "size:", size
            #print "len:", len(trajectories)
            #s1, a, s2, r = random.sample(trajectories, size)
            trajs = random.sample(trajectories, size)
            s1 = []
            s2 = []
            a  = []
            r  = []

            for traj in trajs:
                s1.extend(traj["state"])
                s2.extend(traj["prev_state"])
                a.extend(traj["action"])
                r.extend(traj["reward"])

            #print np.array(s2).shape
            q2 = np.max(self.network.get_q(s2), axis=1)
            q2 = np.array(q2)

            # the value of q2 is ignored in learn if s2 is terminal
            loss = self.network.train(s1, q2, a, r)

            #print q2.mean()
            ############

'''
            '''
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

            # compute validation reward
            print "time limit: ", time_limit
            val_reward = np.array(
                [self.get_trajectory(time_limit, deterministic=True)['reward'].sum() for _ in range(1)]
            )

            # update stats
            mean_train_rs.append(train_rs.mean())
            mean_val_rs.append(val_reward.mean())
            self.loss.append(loss)
'''
            val_reward = np.array(
                [self.get_trajectory(time_limit, deterministic=True)['reward'].sum() for _ in range(1)]
            )
            # print stats
            print '%3d mean reward: %2.2f loss: %2.10f games played: %3d' % (
                epoch + 1, np.mean(val_reward), 00.00, total_games)

            # check for early stopping: true if the validation reward has not changed in n_early_stop epochs
            if early_stop and len(mean_val_rs) >= early_stop and \
                    all([x == mean_val_rs[-1] for x in mean_val_rs[-early_stop:-1]]):
                break

    def get_trajectory(self, time_limit=None, deterministic=True):
        """
        Compute state by iteratively evaluating the agent policy on the environment.
        """
        time_limit = time_limit or self.environment.spec.timestep_limit
        state = self.environment.reset()
        state = self._state_reshape(state)

        prev_state = list(state)

        trajectory = {'state': [], 'action': 0, 'reward': 0, 'prev_state': []}
        baner = []
        #expert = {'state': [], 'action': [], 'reward': [], 'prev_state': []}
        for _ in xrange(time_limit):
            action = self.get_action(state, deterministic)
            #state, reward, done, _ = self.environment.step(action)
            state, reward, done, _ = self.environment.step(action)

            #self.environment.render()

            state = self._state_reshape(state)

            if deterministic:
                #print action
                self.environment.render()

            trajectory['state'] = state
            trajectory['action'] = action
            trajectory['reward'] = reward
            trajectory['prev_state'] = prev_state
            baner.append(trajectory)
            prev_state = list(state)
            if reward == 1:
                print "REWARD"

            size = 64
            if(len(baner)<64):
                size = len(baner)
            #print "size:", size
            #print "len:", len(trajectories)
            #s1, a, s2, r = random.sample(trajectories, size)
            #trajs = sample(baner, size)
            #s1 = []
            #s2 = []
            #a  = []
            #r  = []

            #for traj in trajs:
            #    s1.extend(traj["state"])
            #    s2.extend(traj["prev_state"])
            #    a.extend(traj["action"])
            #    r.extend(traj["reward"])

            #if(len(trajs)!=0):
                #print np.array(s2).shape
            val = np.expand_dims(np.array(state), 0)
            q2 = np.max(self.network.get_q(val), axis=1)
            q2 = np.array(q2)

            val2 = np.expand_dims(np.array(prev_state), 0)
            # the value of q2 is ignored in learn if s2 is terminal
            loss = self.network.train(val2, q2, action, reward)

        #print np.array(trajectory['state']).shape
        return {'state': np.array(trajectory['state']),
                'action': np.array(trajectory['action']),
                'reward': np.array(trajectory['reward']),
                'prev_state': np.array(trajectory['prev_state'])}

    def get_action(self, state, deterministic=True):
        """
        Evaluate the agent policy to choose an action, a, given state, s.
        """
        # compute action probabilities
        #action_probabilities = self.network.evaluate(state.reshape(1, -1))
        action_probabilities = self.network.evaluate(np.expand_dims(state, 0))

        if deterministic:
            # choose action with highest probability
            print action_probabilities
            return action_probabilities.argmax()
        else:
            # sample action from cummulative distribution
            #action = (np.cumsum(np.asarray(action_probabilities)) > np.random.rand()).argmax()
            eps = self.get_exploration_rate()
            if random() <= eps:
                action = randint(0, 5)
            else:
                # Choose the best action according to the network.
                action = action_probabilities.argmax()
                print action
            return action


    def get_exploration_rate(self):
        """# Define exploration rate change over time"""
        epochs = self.epochs
        epoch = self.epoch

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

    def _state_reshape(self, state):
        img = Image.fromarray(state, 'RGB').convert('L')
        size = (self.network.shape[2], self.network.shape[2])
        img.thumbnail(size, Image.ANTIALIAS)
        val = np.expand_dims(np.array(img), 0)
        #val = val.reshape(0, 1, 110, 84)
        return val

Agent.register(AgentPolicy)
