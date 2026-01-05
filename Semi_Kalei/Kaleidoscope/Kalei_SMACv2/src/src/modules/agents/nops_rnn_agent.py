import torch.nn as nn
import torch.nn.functional as F
import torch as th


class NoPSRNNAgent_1R3(nn.Module):
    """
    NoPS (No Parameter Sharing) RNN Agent with 1 RNN and 3 hidden layers.
    Each agent has its own independent set of parameters.
    """
    def __init__(self, input_shape, args):
        super(NoPSRNNAgent_1R3, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # Create independent networks for each agent
        self.agent_networks = nn.ModuleList([
            self._build_agent_network(input_shape, args)
            for _ in range(self.n_agents)
        ])

    def _build_agent_network(self, input_shape, args):
        """Build a single agent's network"""
        return nn.ModuleDict({
            'fc1': nn.Linear(input_shape, args.rnn_hidden_dim),
            'rnn': nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim),
            'fc2': nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            'fc3': nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            'fc4': nn.Linear(args.rnn_hidden_dim, args.n_actions),
        })

    def init_hidden(self):
        """Initialize hidden states for all agents"""
        # Create hidden states for each agent
        hidden_states = []
        for agent_net in self.agent_networks:
            h = agent_net['fc1'].weight.new(1, self.args.rnn_hidden_dim).zero_()
            hidden_states.append(h)
        return th.stack(hidden_states, dim=1)  # (1, n_agents, rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        """
        Forward pass for all agents
        inputs: (batch_size, n_agents, input_size)
        hidden_state: (batch_size, n_agents, rnn_hidden_dim)
        """
        b, a, e = inputs.size()

        # Process each agent independently
        agent_outputs = []
        new_hidden_states = []

        for agent_idx in range(self.n_agents):
            agent_net = self.agent_networks[agent_idx]

            # Get inputs and hidden state for this agent
            agent_input = inputs[:, agent_idx, :]  # (batch_size, input_size)
            agent_h = hidden_state[:, agent_idx, :]  # (batch_size, rnn_hidden_dim)

            # Forward pass through this agent's network
            x = F.relu(agent_net['fc1'](agent_input))
            h_new = agent_net['rnn'](x, agent_h)
            q = F.relu(agent_net['fc2'](h_new))
            q = F.relu(agent_net['fc3'](q))
            q = agent_net['fc4'](q)

            agent_outputs.append(q)
            new_hidden_states.append(h_new)

        # Stack outputs
        q_values = th.stack(agent_outputs, dim=1)  # (batch_size, n_agents, n_actions)
        h_new = th.stack(new_hidden_states, dim=1)  # (batch_size, n_agents, rnn_hidden_dim)

        return q_values, h_new


class NoPSRNNAgent(nn.Module):
    """
    NoPS (No Parameter Sharing) RNN Agent (simpler version).
    Each agent has its own independent set of parameters.
    """
    def __init__(self, input_shape, args):
        super(NoPSRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # Create independent networks for each agent
        self.agent_networks = nn.ModuleList([
            self._build_agent_network(input_shape, args)
            for _ in range(self.n_agents)
        ])

    def _build_agent_network(self, input_shape, args):
        """Build a single agent's network"""
        return nn.ModuleDict({
            'fc1': nn.Linear(input_shape, args.rnn_hidden_dim),
            'rnn': nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim),
            'fc2': nn.Linear(args.rnn_hidden_dim, args.n_actions),
        })

    def init_hidden(self):
        """Initialize hidden states for all agents"""
        hidden_states = []
        for agent_net in self.agent_networks:
            h = agent_net['fc1'].weight.new(1, self.args.rnn_hidden_dim).zero_()
            hidden_states.append(h)
        return th.stack(hidden_states, dim=1)

    def forward(self, inputs, hidden_state):
        """
        Forward pass for all agents
        inputs: (batch_size, n_agents, input_size)
        hidden_state: (batch_size, n_agents, rnn_hidden_dim)
        """
        b, a, e = inputs.size()

        # Process each agent independently
        agent_outputs = []
        new_hidden_states = []

        for agent_idx in range(self.n_agents):
            agent_net = self.agent_networks[agent_idx]

            # Get inputs and hidden state for this agent
            agent_input = inputs[:, agent_idx, :]
            agent_h = hidden_state[:, agent_idx, :]

            # Forward pass through this agent's network
            x = F.relu(agent_net['fc1'](agent_input))
            h_new = agent_net['rnn'](x, agent_h)
            q = agent_net['fc2'](h_new)

            agent_outputs.append(q)
            new_hidden_states.append(h_new)

        # Stack outputs
        q_values = th.stack(agent_outputs, dim=1)
        h_new = th.stack(new_hidden_states, dim=1)

        return q_values, h_new
