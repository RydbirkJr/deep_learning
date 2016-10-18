from src.agent.agent import Agent
import numpy as np


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
        loss = []

        for epoch in xrange(epochs):

            # 1. collect trajectories until we have at least states_per_batch total timesteps
            trajectories = []
            total_trajectories = 0
            while total_trajectories < states_per_batch:
                trajectory = self.get_trajectory(time_limit, deterministic=False)
                trajectories.append(trajectory)
                total_trajectories += len(trajectory["reward"])

            all_states = np.concatenate([trajectory["state"] for trajectory in trajectories])

            # 2. compute cumulative discounted rewards (returns)
            rets = [self._cumulative_discount(trajectory["reward"], discount_factor) for trajectory in trajectories]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

            # 3. compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)

            # 4. compute advantages
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_actions = np.concatenate([trajectory["action"] for trajectory in trajectories])
            all_advantages = np.concatenate(advs)

            # 5. do policy gradient update step
            loss = self.network.train(all_states, all_actions, all_advantages, learning_rate)
            train_rs = np.array([trajectory["reward"].sum() for trajectory in trajectories])  # trajectory total rewards
            eplens = np.array([len(trajectory["reward"]) for trajectory in trajectories])  # trajectory lengths

            # compute validation reward
            val_rs = np.array(
                [self.get_trajectory(time_limit, deterministic=True)['reward'].sum() for _ in range(10)]
            )

            # update stats
            mean_train_rs.append(train_rs.mean())
            mean_val_rs.append(val_rs.mean())
            loss.append(loss)

            # print stats
            print '%3d mean_train_r: %6.2f mean_val_r: %6.2f loss: %f' % (
                epoch + 1, train_rs.mean(), val_rs.mean(), loss)

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

        trajectory = {'state': [], 'action': [], 'reward': []}

        for _ in xrange(time_limit):
            action = self.get_action(state, deterministic)
            (state, reward, done, _) = self.environment.step(action)

            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)

            if done: break

        return {'state': np.array(trajectory['state']),
                'action': np.array(trajectory['action']),
                'reward': np.array(trajectory['reward'])}

    def get_action(self, state, deterministic=True):
        """
        Evaluate the agent policy to choose an action, a, given state, s.
        """
        # compute action probabilities
        action_probabilities = self.network.evaluate(state.reshape(1, -1))

        if deterministic:
            # choose action with highest probability
            return action_probabilities.argmax()
        else:
            # sample action from cummulative distribution
            return (np.cumsum(np.asarray(action_probabilities)) > np.random.rand()).argmax()

    def _cumulative_discount(self, reward, gamma):
        """
        Compute the cumulative discounted rewards.
        """
        reward_out = np.zeros(len(reward), 'float64')
        reward_out[-1] = reward[-1]
        for i in reversed(xrange(len(reward) - 1)):
            reward_out[i] = reward[i] + gamma * reward_out[i + 1]
        return reward_out

Agent.register(AgentPolicy)
