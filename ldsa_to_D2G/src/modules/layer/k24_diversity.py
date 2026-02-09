"""
K-2:4 Pattern Diversity Loss Module.

This module implements Module C: Pattern Orthogonality Diversity Loss.
This encourages different agents to select different sparse patterns for heterogeneity.

References:
- 最终方案设计_K_24.md
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PatternOrthogonalityLoss(nn.Module):
    """
    Module C: Pattern Orthogonality Diversity Loss.

    Encourages different agents to select different 2:4 sparse patterns
    by minimizing the dot product between their pattern probability distributions.

    Loss = mean(pi_a * pi_b) for all pairs of agents a, b

    Args:
        n_agents: Number of agents
        reduction: How to reduce the loss ('mean', 'sum', 'none')
    """

    def __init__(self, n_agents, reduction='mean'):
        super(PatternOrthogonalityLoss, self).__init__()
        self.n_agents = n_agents
        self.reduction = reduction

    def forward(self, pattern_probs_list):
        """
        Compute pattern orthogonality loss.

        Args:
            pattern_probs_list: List of [batch_size, n_groups, 6] pattern probabilities
                               for each agent, or a single tensor [batch, n_agents, n_groups, 6]

        Returns:
            loss: Scalar loss value
        """
        # Handle different input formats
        if isinstance(pattern_probs_list, list):
            # List of tensors per agent
            if len(pattern_probs_list) < 2:
                return th.tensor(0.0, device=pattern_probs_list[0].device)

            # Stack: [n_agents, batch, n_groups, 6]
            stacked = th.stack(pattern_probs_list, dim=0)
            # Permute to [batch, n_agents, n_groups, 6]
            stacked = stacked.permute(1, 0, 2, 3)
        else:
            # Single tensor: [batch, n_agents, n_groups, 6]
            stacked = pattern_probs_list

        if stacked.shape[1] < 2:
            return th.tensor(0.0, device=stacked.device)

        batch_size, n_agents, n_groups, n_patterns = stacked.shape

        # Compute pairwise dot products
        # For efficiency, compute all pairs at once
        loss_values = []

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Dot product of probability distributions
                # [batch, n_groups, 6] * [batch, n_groups, 6] -> [batch, n_groups]
                dot_product = (stacked[:, i, :, :] * stacked[:, j, :, :]).sum(dim=-1)
                loss_values.append(dot_product)

        # Stack and average
        # [n_pairs, batch, n_groups]
        loss_tensor = th.stack(loss_values, dim=0)

        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        else:
            return loss_tensor


class K24DiversityManager:
    """
    Manages diversity loss with adaptive coefficient.

    Adjusts the diversity loss coefficient based on the ratio of
    TD loss to diversity loss to maintain balanced training.

    Args:
        base_coef: Base diversity coefficient
        target_ratio: Target ratio of div_loss / td_loss
        min_coef: Minimum coefficient
        max_coef: Maximum coefficient
        adjustment_factor: Factor for coefficient adjustment
    """

    def __init__(self, base_coef=0.1, target_ratio=1.0, min_coef=0.01, max_coef=1.0, adjustment_factor=1.1):
        self.base_coef = base_coef
        self.target_ratio = target_ratio
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.adjustment_factor = adjustment_factor
        self.current_coef = base_coef

    def update_coef(self, td_loss, div_loss):
        """
        Update diversity coefficient based on loss ratio.

        Args:
            td_loss: Current TD loss value
            div_loss: Current diversity loss value
        """
        if div_loss > 0:
            ratio = td_loss / (div_loss + 1e-10)

            if ratio > self.target_ratio:
                # TD loss dominates, increase diversity coefficient
                self.current_coef = min(
                    self.current_coef * self.adjustment_factor,
                    self.max_coef
                )
            else:
                # Diversity loss dominates, decrease coefficient
                self.current_coef = max(
                    self.current_coef / self.adjustment_factor,
                    self.min_coef
                )

        return self.current_coef

    def get_coef(self):
        """Get current diversity coefficient."""
        return self.current_coef

    def reset(self):
        """Reset to base coefficient."""
        self.current_coef = self.base_coef


class LayerPatternTracker:
    """
    Helper class to track pattern probabilities per layer.

    Useful for analysis and debugging of pattern selection dynamics.

    Args:
        layer_names: List of layer names to track
    """

    def __init__(self, layer_names=None):
        self.layer_names = layer_names or []
        self.pattern_history = {name: [] for name in self.layer_names}

    def update(self, layer_name, pattern_probs):
        """
        Update pattern statistics for a layer.

        Args:
            layer_name: Name of the layer
            pattern_probs: [batch, n_groups, 6] pattern probabilities
        """
        if layer_name not in self.pattern_history:
            self.pattern_history[layer_name] = []

        # Average over batch and groups to get distribution
        dist = pattern_probs.mean(dim=(0, 1)).detach().cpu()
        self.pattern_history[layer_name].append(dist)

    def get_pattern_distribution(self, layer_name, last_n=None):
        """
        Get average pattern distribution for a layer.

        Args:
            layer_name: Name of the layer
            last_n: If specified, average over last n updates

        Returns:
            distribution: [6] average pattern distribution
        """
        if layer_name not in self.pattern_history:
            return None

        history = self.pattern_history[layer_name]
        if last_n is not None:
            history = history[-last_n:]

        if not history:
            return None

        return th.stack(history).mean(dim=0)

    def get_sparsity_report(self):
        """
        Generate a report of pattern usage statistics.

        Returns:
            report: Dictionary with pattern statistics per layer
        """
        report = {}
        for name, history in self.pattern_history.items():
            if not history:
                continue

            # Stack all history
            all_probs = th.stack(history)  # [n_updates, 6]

            report[name] = {
                'avg_distribution': all_probs.mean(dim=0).tolist(),
                'std_distribution': all_probs.std(dim=0).tolist(),
                'dominant_pattern': int(all_probs.mean(dim=0).argmax().item()),
                'n_updates': len(history)
            }

        return report
