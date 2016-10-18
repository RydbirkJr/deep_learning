from abc import abstractmethod, ABCMeta

class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_action(self):
        pass