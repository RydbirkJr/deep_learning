from agent import Agent
import numpy as np
from PIL import Image
import random


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

        for epoch in xrange(epochs):

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
            size = 1000
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
                epoch + 1, np.mean(val_reward), loss, total_games)

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

        trajectory = {'state': [], 'action': [], 'reward': [], 'prev_state': []}

        for _ in xrange(time_limit):
            action = self.get_action(state, deterministic)
            #state, reward, done, _ = self.environment.step(action)
            for i in range(3):
                state, reward, done, _ = self.environment.step(action)
                if reward != 0:
                    break;

            #self.environment.render()

            state = self._state_reshape(state)

            if deterministic:
                #print action
                self.environment.render()

            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward+0.001)
            trajectory['prev_state'].append(prev_state)

            prev_state = list(state)
            if reward == 1:
                print "REWARD"
            if done:
                break
            if reward == -1:
                break

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
            action = (np.cumsum(np.asarray(action_probabilities)) > np.random.rand()).argmax()
            #print action
            return action

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
        return state.reshape(3,210,160)
        img = Image.fromarray(state, 'RGB').convert('L')
        size = (self.network.shape[3], self.network.shape[2])
        img.thumbnail(size, Image.ANTIALIAS)
        print np.array(img).shape
        val = np.expand_dims(np.array(img), 3)
        #val = val.reshape(0, 1, 110, 84)
        return val

Agent.register(AgentPolicy)
