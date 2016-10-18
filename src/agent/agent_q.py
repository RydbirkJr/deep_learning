from src.agent.agent import Agent


class AgentQ(Agent):

    def __init__(self, environment, network):
        super(AgentQ, self).__init__(environment, network)

Agent.register(AgentQ)
