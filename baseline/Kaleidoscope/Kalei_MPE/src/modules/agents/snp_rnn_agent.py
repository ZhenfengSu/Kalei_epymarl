import torch.nn as nn
import torch.nn.functional as F
import torch as th


class SNP_RNNAgent_1R3(nn.Module):
    """
    SNP (Structured Network Pruning) Agent implementation for MPE.

    This agent implements the SNP-PS (Structured Network Pruning for Parameter Sharing) method
    based on the Lottery Group Ticket Hypothesis. It uses structured pruning to create
    subnetworks for different agents within a shared root network.

    Key mechanisms:
    1. Each agent has its own binary mask for each layer
    2. Masks are randomly generated based on sparsity ratios
    3. Masks create partially shared and partially independent parameters
    4. Overlapping regions enable parameter sharing (sample efficiency)
    5. Non-overlapping regions provide agent-specific capacity (representational capacity)

    Differences from SMACv2 version:
    - Uses n_agents instead of n_unit_types
    - Uses agent_ids directly (0 to n_agents-1) instead of unit type extraction
    """

    def __init__(self, input_shape, args):
        super(SNP_RNNAgent_1R3, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # Get SNP-specific arguments
        self.sparsity_ratios = self.args.SNP_args["layers_sparsities"]

        # Ensure sparsity ratios are provided for all 3 layers (fc2, fc3, fc4)
        assert len(self.sparsity_ratios) == 3, (
            f"Expected 3 sparsity ratios for layers fc2, fc3, fc4, "
            f"got {len(self.sparsity_ratios)}"
        )

        # Standard network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc4 = nn.Linear(args.hidden_dim, args.n_actions)

        # Generate and register binary masks for each layer
        # Masks are registered as buffers to be part of state_dict but not parameters
        for i, layer_sparsity in enumerate(self.sparsity_ratios):
            self.register_buffer(
                f"mask_{i}",
                th.rand(self.n_agents, args.hidden_dim) > layer_sparsity,
            )
            # Convert boolean to float (0.0 or 1.0)
            setattr(self, f"mask_{i}", getattr(self, f"mask_{i}").float())

    def init_hidden(self):
        """Initialize hidden state for RNN."""
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, agent_ids):
        """
        Forward pass with structured network pruning.

        Args:
            inputs: Input observations [batch_size, n_agents, input_dim]
            hidden_state: RNN hidden states [batch_size, n_agents, hidden_dim]
            agent_ids: Agent IDs [batch_size, n_agents]

        Returns:
            q: Q-values [batch_size, n_agents, n_actions]
            h: Updated hidden states [batch_size, n_agents, hidden_dim]
        """
        # Get dimensions
        b, a, e = inputs.size()

        # Reshape inputs and agent_ids for processing
        inputs = inputs.view(-1, e)
        agent_ids = agent_ids.reshape(-1)

        # First layer: standard (no mask)
        x = F.relu(self.fc1(inputs))

        # RNN layer: standard (no mask)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

        # Apply structured pruning masks to subsequent layers
        # Although not explicitly specified in the SNP paper,
        # structured pruning typically happens before activation

        # Layer fc2 with mask_0
        h = h * self.mask_0[agent_ids]
        q = self.fc2(h)
        q = F.relu(q)

        # Layer fc3 with mask_1
        q = q * self.mask_1[agent_ids]
        q = self.fc3(q)
        q = F.relu(q)

        # Layer fc4 with mask_2
        q = q * self.mask_2[agent_ids]
        q = self.fc4(q)

        # Reshape outputs
        return q.view(b, a, -1), h.view(b, a, -1)

    def get_sparsities(self):
        """
        Calculate the actual sparsity ratios for each masked layer.

        Returns:
            sparsities: List of sparsity ratios for each masked layer
            mask_stats: Dictionary with detailed mask statistics
        """
        sparsities = []
        mask_stats = {}

        for i in range(3):  # We have 3 masks
            mask = getattr(self, f"mask_{i}")
            # Calculate actual sparsity (ratio of zeros)
            actual_sparsity = 1.0 - mask.mean().item()
            sparsities.append(actual_sparsity)

            # Calculate overlap statistics
            # How many neurons are shared by at least k agents?
            n_agents = mask.size(0)
            n_neurons = mask.size(1)

            # Count how many agents use each neuron
            usage_count = mask.sum(dim=0)  # [n_neurons]

            # Overlap distribution
            overlap_dist = {}
            for k in range(1, n_agents + 1):
                n_neurons_with_k_agents = (usage_count == k).sum().item()
                overlap_dist[f"used_by_{k}_agents"] = n_neurons_with_k_agents

            mask_stats[f"layer_{i}"] = {
                "target_sparsity": self.sparsity_ratios[i],
                "actual_sparsity": actual_sparsity,
                "overlap_distribution": overlap_dist,
                "n_neurons": n_neurons,
            }

        return sparsities, mask_stats

    def get_mask_diversity(self):
        """
        Calculate mask diversity metrics.

        Returns:
            diversity_metrics: Dictionary with diversity statistics
        """
        diversity_metrics = {}

        for i in range(3):
            mask = getattr(self, f"mask_{i}")
            n_agents = mask.size(0)

            # Calculate pairwise differences between masks
            # This measures how different each agent's mask is from others
            pairwise_diff = 0.0
            for j in range(n_agents):
                for k in range(j + 1, n_agents):
                    # Hamming distance: proportion of differing positions
                    diff = (mask[j] != mask[k]).float().mean().item()
                    pairwise_diff += diff

            # Average pairwise difference
            n_pairs = n_agents * (n_agents - 1) / 2
            avg_pairwise_diff = pairwise_diff / n_pairs if n_pairs > 0 else 0.0

            diversity_metrics[f"layer_{i}"] = {
                "avg_pairwise_difference": avg_pairwise_diff,
                "expected_random_diff": self.sparsity_ratios[i] * (1 - self.sparsity_ratios[i]) * 2,
            }

        return diversity_metrics
