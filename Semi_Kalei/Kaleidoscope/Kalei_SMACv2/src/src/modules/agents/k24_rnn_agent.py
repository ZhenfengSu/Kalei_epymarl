"""
K-2:4 RNN Agent with Semi-structured Sparsity for SMACv2

Implements the full K-2:4 algorithm for StarCraft II multi-agent scenarios.
Adapted from the epymarl implementation with SMACv2-specific adjustments.
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from ..layer.k24_pattern_gumbel_layer import SemiStructuredLinear24
from ..layer.k24_diversity import LayerPatternTracker


class K24_type_NRNNAgent_1R3(nn.Module):
    """
    K-2:4 Type-specific RNN Agent for SMACv2.

    In SMACv2, agents belong to different unit types (e.g., Marine, Marauder, Medivac).
    This agent creates heterogeneous strategies for different unit types using 2:4
    semi-structured sparsity.

    Key differences from base K24_RNNAgent:
    - Uses n_unit_types instead of n_agents
    - Adapted for SMACv2 controller structure

    Args:
        input_shape: Input feature dimension
        args: Configuration with:
            - n_unit_types: Number of unit types (e.g., 3 for Marine/Marauder/Medivac)
            - rnn_hidden_dim: Hidden dimension
            - n_actions: Number of actions
            - K24_args: K-2:4 specific parameters
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_unit_types  # Unit types in SMACv2

        # Extract K-2:4 arguments
        k24_args = getattr(args, 'K24_args', {})
        self.temperature_init = k24_args.get("temperature_init", 5.0)
        self.temperature_min = k24_args.get("temperature_min", 0.1)
        self.ema_momentum = k24_args.get("ema_momentum", 0.99)
        self.hetero_init_scale = k24_args.get("hetero_init_scale", 0.01)
        self.anneal_start = k24_args.get("anneal_start", 0.0)
        self.anneal_end = k24_args.get("anneal_end", 0.8)

        # Create K-2:4 Linear layer factory
        self.K24_linear = partial(
            SemiStructuredLinear24,
            n_agents=self.n_agents,
            hidden_dim=args.rnn_hidden_dim,
            temperature_init=self.temperature_init,
            temperature_min=self.temperature_min,
            ema_momentum=self.ema_momentum,
            hetero_init_scale=self.hetero_init_scale,
        )

        # Build 1R3 architecture (1 recurrent + 3 hidden layers)
        self.fc1 = self.K24_linear(
            in_features=input_shape,
            out_features=args.rnn_hidden_dim
        )
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = self.K24_linear(
            in_features=args.rnn_hidden_dim,
            out_features=args.rnn_hidden_dim
        )
        self.fc3 = self.K24_linear(
            in_features=args.rnn_hidden_dim,
            out_features=args.rnn_hidden_dim
        )
        self.fc4 = self.K24_linear(
            in_features=args.rnn_hidden_dim,
            out_features=args.n_actions
        )

        # Layers with 2:4 sparsity
        self.mask_layers = [self.fc1, self.fc2, self.fc3, self.fc4]

        # Layers eligible for resetting
        self.reset_layers = [self.fc2, self.fc3, self.fc4]

        # Pattern tracker for diversity loss
        self.pattern_tracker = LayerPatternTracker()
        for layer in self.mask_layers:
            self.pattern_tracker.register_layer(layer)

    def init_hidden(self):
        """Initialize hidden states for RNN."""
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, agent_ids):
        """
        Forward pass with unit-type-specific 2:4 semi-structured sparse masks.

        Args:
            inputs: [batch_size, n_agents, input_dim]
            hidden_state: [batch_size, n_agents, hidden_dim]
            agent_ids: [batch_size, n_agents] unit type indices

        Returns:
            q: [batch_size, n_agents, n_actions]
            h: [batch_size, n_agents, hidden_dim]
        """
        # Reshape inputs
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        agent_ids = agent_ids.reshape(-1)

        # Forward through layers with unit-type-specific 2:4 masks
        x = F.relu(self.fc1(inputs, agent_ids))

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        q = F.relu(self.fc2(h, agent_ids))
        q = F.relu(self.fc3(q, agent_ids))
        q = self.fc4(q, agent_ids)

        return q.view(b, a, -1), h.view(b, a, -1)

    def set_require_grads(self, mode):
        """Enable/disable gradient computation for heterogeneity parameters."""
        assert mode in [True, False], f"Invalid mode {mode}."
        for layer in self.mask_layers:
            layer.hetero_alpha.requires_grad = mode

    def get_pattern_probs(self):
        """Collect pattern probabilities from all layers."""
        return self.pattern_tracker.collect_patterns()

    def _get_linear_weight_sparsities(self):
        """Calculate sparsity for each layer."""
        sparsities = []
        w_counts = []
        for layer in self.mask_layers:
            sparsity = layer.get_sparsity()
            sparsities.append(sparsity)
            # Estimate weight count
            w_counts.append(layer.weight.numel())
        return th.tensor(sparsities), w_counts

    def get_sparsities(self):
        """
        Calculate overall sparsity statistics.

        Returns:
            layer_sparsities: Mean sparsity per layer
            layer_sparsities_var: Variance (zero for fixed 2:4)
            overall_sparsity: Total network sparsity
        """
        # Calculate total parameter count
        total_params = sum(p.numel() for n, p in self.named_parameters()
                          if "hetero_alpha" not in n and "temperature" not in n)

        # Calculate sparsity per layer
        w_sparsities, w_counts = self._get_linear_weight_sparsities()
        w_sparsities_var = th.zeros_like(w_sparsities)  # Fixed sparsity for 2:4

        zero_params = 0
        for sparsity, w_count in zip(w_sparsities, w_counts):
            zero_params += w_count * sparsity.item()

        overall_sparsity = zero_params / total_params

        return w_sparsities, w_sparsities_var, overall_sparsity

    def mask_diversity_loss(self):
        """
        Placeholder for diversity loss (computed in learner).

        The actual pattern orthogonality loss is computed in K24_NQLearner
        using K24DiversityManager for better control.
        """
        return th.tensor(0.0, device=self.fc1.weight.device)

    def anneal_temperature(self, progress):
        """
        Anneal temperature for all layers.

        Args:
            progress: Training progress (0.0 to 1.0)
        """
        if progress < self.anneal_start:
            temp = self.temperature_init
        elif progress > self.anneal_end:
            temp = self.temperature_min
        else:
            normalized_progress = (progress - self.anneal_start) / (self.anneal_end - self.anneal_start)
            temp = self.temperature_init - (self.temperature_init - self.temperature_min) * normalized_progress

        for layer in self.mask_layers:
            layer.set_temperature(temp)

    def _reset_all_masks_weights(self, reset_ratio=0.1):
        """Reset heterogeneity coefficients for exploration."""
        for layer in self.reset_layers:
            layer.reset_hetero_alpha(reset_ratio)

    @property
    def mask_parameters(self):
        """Return heterogeneity parameters."""
        return [l.hetero_alpha for l in self.mask_layers]
