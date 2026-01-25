# SNP RNN Agent with structured pruning for heterogeneity
# Based on the Structured Network Pruning for Parameter Sharing (SNP-PS) algorithm
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SNP_RNNAgent_1R3(nn.Module):
    """
    RNN Agent with SNP (Structured Network Pruning) heterogeneous masks.

    Uses fixed random binary masks for parameter sharing across agents.
    Masks are generated at initialization and remain static during training.

    Architecture: 1R3 (1 recurrent/RNN layer + 3 hidden layers with masks)
    """
    def __init__(self, input_shape, args):
        super(SNP_RNNAgent_1R3, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # Get sparsity ratios for each masked layer
        self.sparsity_ratios = args.SNP_args["layers_sparsities"]
        assert len(self.sparsity_ratios) == 3, "SNP requires exactly 3 sparsity ratios for fc2, fc3, fc4"

        # Standard input layer (no masking)
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)

        # Recurrent layer (no masking)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

        # Output layers with structured pruning masks
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc4 = nn.Linear(args.hidden_dim, args.n_actions)

        # Generate and register binary masks for each agent
        # mask_0 for fc2, mask_1 for fc3, mask_2 for fc4
        for i, layer_sparsity in enumerate(self.sparsity_ratios):
            self.register_buffer(
                f"mask_{i}",
                th.rand(self.n_agents, args.hidden_dim) > layer_sparsity,
            )

    def init_hidden(self):
        """Initialize hidden states."""
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, agent_ids):
        """
        Forward pass with agent-specific structured pruning masks.

        Args:
            inputs: Agent inputs [batch_size * n_agents, input_dim]
            hidden_state: Hidden states from previous timestep [batch_size * n_agents, hidden_dim]
            agent_ids: Agent IDs for mask selection [batch_size * n_agents]

        Returns:
            q: Q-values [batch_size, n_agents, n_actions]
            h: Updated hidden states [batch_size, n_agents, hidden_dim]
        """
        # Reshape inputs
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        agent_ids = agent_ids.reshape(-1)

        # Forward through fc1 (no mask)
        x = F.relu(self.fc1(inputs))

        # Forward through RNN (no mask)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

        # Apply structured pruning masks after RNN
        # Masks are applied to activations before the linear transformation
        h = h * self.mask_0[agent_ids].float()

        # Forward through fc2 with mask_1
        q = self.fc2(h) * self.mask_1[agent_ids].float()
        q = F.relu(q)

        # Forward through fc3 with mask_2
        q = self.fc3(q) * self.mask_2[agent_ids].float()
        q = F.relu(q)

        # Forward through fc4 (output layer, no additional mask needed)
        q = self.fc4(q)

        return q.view(b, a, -1), h.view(b, a, -1)

    def get_sparsities(self):
        """
        Calculate actual sparsity ratios for each layer's masks.

        Returns:
            sparsities: List of sparsity ratios for each masked layer
            mask_counts: Number of neurons in each masked layer
        """
        sparsities = []
        mask_counts = []

        for i in range(3):
            mask = getattr(self, f"mask_{i}")
            # Calculate sparsity: fraction of False (pruned) neurons
            sparsity = (1.0 - mask.float().mean(dim=-1)).mean().item()
            count = mask.numel() // self.n_agents  # neurons per agent
            sparsities.append(sparsity)
            mask_counts.append(count)

        return sparsities, mask_counts

    def get_mask_diversity(self):
        """
        Measure pairwise differences between agent masks.

        Higher diversity means agents have more different active neurons,
        which encourages agent specialization and heterogeneity.

        Returns:
            diversity_score: Average pairwise mask difference across all layers
        """
        total_diversity = 0.0

        for i in range(3):
            mask = getattr(self, f"mask_{i}")
            # Calculate pairwise differences
            # mask shape: [n_agents, hidden_dim]
            mask_expanded_1 = mask.unsqueeze(1)  # [n_agents, 1, hidden_dim]
            mask_expanded_2 = mask.unsqueeze(0)  # [1, n_agents, hidden_dim]

            # Count differing neurons between each pair of agents
            differences = (mask_expanded_1 != mask_expanded_2).float()
            diversity = differences.mean().item()
            total_diversity += diversity

        return total_diversity / 3.0

    def get_overlap_statistics(self):
        """
        Analyze parameter sharing patterns across agents.

        Returns:
            overlap_matrix: Matrix showing fraction of shared neurons between each agent pair
            layer_overlaps: Overlap statistics for each layer
        """
        overlap_matrix = th.zeros(self.n_agents, self.n_agents)
        layer_overlaps = []

        for i in range(3):
            mask = getattr(self, f"mask_{i}")
            # Calculate overlap: fraction of neurons both agents have active
            mask_float = mask.float()
            overlap = (mask_float @ mask_float.T) / mask_float.sum(dim=-1, keepdim=True)
            layer_overlaps.append(overlap.clone())
            overlap_matrix += overlap

        overlap_matrix /= 3.0
        return overlap_matrix, layer_overlaps

    def get_active_neuron_counts(self):
        """
        Get the number of active (non-pruned) neurons for each agent in each layer.

        Returns:
            active_counts: List of [n_agents] arrays with active neuron counts
        """
        active_counts = []

        for i in range(3):
            mask = getattr(self, f"mask_{i}")
            counts = mask.sum(dim=-1)  # [n_agents]
            active_counts.append(counts)

        return active_counts
