from abc import abstractmethod, ABCMeta


class Agent(object):
    __metaclass__ = ABCMeta

    # network = None
    # environment = None

    def __init__(self, environment, network):
        # type: (EnvSpec, Network) -> Agent
        """
        Initialises agent work environment and network
        :param environment: EnvSpec
        :param network: Network
        """
        self.environment = environment
        self.network = network

    @abstractmethod
    def learn(self,
              epochs=100,
              states_per_batch=1000,
              time_limit=None,
              learning_rate=0.001,
              discount_factor=1.0,
              early_stop=None):
        raise NotImplementedError()

    @abstractmethod
    def get_trajectory(self, epoch, epochs, time_limit=None, deterministic=True):
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, epoch, epochs, state, deterministic=True):
        raise NotImplementedError()

