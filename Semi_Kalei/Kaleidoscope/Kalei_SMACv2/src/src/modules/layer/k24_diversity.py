"""
K-2:4 Pattern Orthogonality Diversity Loss

Implements Module C: Pattern Orthogonality (Heterogeneity Enforcement)

This module enforces agent diversity by encouraging different agents to select
different 2:4 sparse patterns. Instead of comparing specific weight values,
we compare pattern probability distributions directly.

Key Innovation:
- Compare pattern selection distributions rather than weight masks
- Minimize dot product between distributions (orthogonality)
- More direct and interpretable than L1 distance on masks
"""

import torch as th
import torch.nn as nn


class PatternOrthogonalityLoss(nn.Module):
    """
    Module C: Pattern Orthogonality Diversity Loss

    Encourages agents to select different 2:4 sparse patterns by minimizing
    the dot product of their pattern probability distributions.

    Mathematical formulation:
        L_div = (1/N_groups) * sum_g (pi_A,g * pi_B,g)

    Where:
        pi_i,g: Softmax probability distribution over 6 patterns for agent i, group g
        Minimizing dot product encourages orthogonality (different patterns)

    Args:
        n_agents: Number of agents
        normalize: Whether to normalize loss by number of agent pairs
    """

    def __init__(self, n_agents, normalize=True):
        super().__init__()
        self.n_agents = n_agents
        self.normalize = normalize

    def forward(self, pattern_probs_list):
        """
        Compute pattern orthogonality loss across all agents.

        Args:
            pattern_probs_list: List of [n_agents, ...] tensors
                Each tensor contains pattern probabilities for one layer
                Shape after reshape: [n_agents, n_groups, 6]

        Returns:
            loss: Scalar diversity loss (lower = more diverse)
            stats: Dictionary with auxiliary statistics
        """
        if not pattern_probs_list:
            return th.tensor(0.0, device='cpu'), {}

        losses = []
        all_pairwise_similarities = []

        # Process each layer's pattern probabilities
        for pattern_probs in pattern_probs_list:
            if pattern_probs is None:
                continue

            # pattern_probs: [batch, n_agents, n_groups, 6] or [n_agents, n_groups, 6]
            # Average over batch dimension if present
            if pattern_probs.dim() == 4:
                pattern_probs = pattern_probs.mean(dim=0)  # [n_agents, n_groups, 6]

            n_agents_in_batch = pattern_probs.shape[0]
            n_groups = pattern_probs.shape[1]

            # Compute pairwise orthogonality loss
            # For each group, compute dot product between all agent pairs
            group_losses = []

            for g in range(n_groups):
                # pattern_probs[:, g, :]: [n_agents, 6]
                pi = pattern_probs[:, g, :]  # [n_agents, 6]

                # Compute all pairwise dot products
                # [n_agents, 6] @ [6, n_agents] -> [n_agents, n_agents]
                similarity_matrix = th.matmul(pi, pi.T)

                # Extract upper triangle (excluding diagonal)
                mask = th.triu(th.ones_like(similarity_matrix), diagonal=1).bool()
                pairwise_similarities = similarity_matrix[mask]

                group_losses.append(pairwise_similarities.mean())
                all_pairwise_similarities.append(pairwise_similarities)

            # Average over groups
            if group_losses:
                losses.append(th.stack(group_losses).mean())

        if not losses:
            return th.tensor(0.0, device='cpu'), {}

        # Average over layers
        loss = th.stack(losses).mean()

        # Normalize by number of agent pairs
        if self.normalize:
            n_pairs = self.n_agents * (self.n_agents - 1) / 2
            loss = loss / max(n_pairs, 1)

        # Compute statistics
        stats = {}
        if all_pairwise_similarities:
            all_sims = th.cat(all_pairwise_similarities)
            stats['mean_similarity'] = all_sims.mean().item()
            stats['max_similarity'] = all_sims.max().item()
            stats['min_similarity'] = all_sims.min().item()
            stats['std_similarity'] = all_sims.std().item()

        return loss, stats


class K24DiversityManager:
    """
    Manages diversity loss computation and adaptive coefficient scheduling.

    Features:
    1. Pattern orthogonality loss computation
    2. Adaptive diversity coefficient based on loss history
    3. Logging and statistics tracking
    """

    def __init__(self, n_agents, base_div_coef=0.1, deque_len=100):
        self.n_agents = n_agents
        self.base_div_coef = base_div_coef

        # Loss history for adaptive coefficient
        from collections import deque
        self.td_loss_history = deque(maxlen=deque_len)
        self.div_loss_history = deque(maxlen=deque_len)

        # Diversity loss module
        self.diversity_loss = PatternOrthogonalityLoss(n_agents, normalize=True)

        # Statistics tracking
        self.stats = {}

    def compute_loss(self, pattern_probs_list, td_loss):
        """
        Compute diversity loss and adaptive coefficient.

        Args:
            pattern_probs_list: List of pattern probability tensors from all layers
            td_loss: Current TD loss (for adaptive coefficient)

        Returns:
            div_loss: Diversity loss value
            div_coef: Adaptive diversity coefficient
            stats: Statistics dictionary
        """
        # Compute pattern orthogonality loss
        div_loss, stats = self.diversity_loss(pattern_probs_list)

        # Update history
        self.td_loss_history.append(td_loss.item() if th.is_tensor(td_loss) else td_loss)
        self.div_loss_history.append(div_loss.item() if th.is_tensor(div_loss) else div_loss)

        # Adaptive diversity coefficient
        # Scale based on ratio of TD loss to diversity loss
        from statistics import mean
        if len(self.div_loss_history) > 0 and mean(self.div_loss_history) > 0:
            div_coef = abs(
                self.base_div_coef * mean(self.td_loss_history) / mean(self.div_loss_history)
            )
        else:
            div_coef = self.base_div_coef

        self.stats = stats
        self.stats['div_coef'] = div_coef

        return div_loss, div_coef, stats

    def get_stats(self):
        """Return latest statistics."""
        return self.stats


class LayerPatternTracker:
    """
    Helper class to track pattern probabilities from all layers.

    Used to collect pattern probabilities from multiple K24Linear layers
    for computing diversity loss.
    """

    def __init__(self):
        self.pattern_probs_list = []
        self.layers = []

    def register_layer(self, layer):
        """Register a K24Linear layer for tracking."""
        self.layers.append(layer)

    def collect_patterns(self):
        """Collect pattern probabilities from all registered layers."""
        self.pattern_probs_list = []
        for layer in self.layers:
            probs = layer.get_pattern_probs()
            if probs is not None:
                self.pattern_probs_list.append(probs)
        return self.pattern_probs_list

    def get_pattern_stats(self):
        """
        Get detailed statistics about pattern usage.

        Returns:
            stats: Dictionary with pattern distribution statistics
        """
        if not self.layers:
            return {}

        all_pattern_dists = []
        for layer in self.layers:
            dist = layer.get_pattern_distribution()
            if dist is not None:
                all_pattern_dists.append(dist.cpu().numpy())

        if not all_pattern_dists:
            return {}

        import numpy as np
        all_pattern_dists = np.array(all_pattern_dists)  # [n_layers, 6]

        stats = {
            'pattern_mean': all_pattern_dists.mean(axis=0).tolist(),
            'pattern_std': all_pattern_dists.std(axis=0).tolist(),
            'pattern_entropy': self._compute_entropy(all_pattern_dists.mean(axis=0))
        }

        return stats

    @staticmethod
    def _compute_entropy(prob_dist):
        """Compute entropy of probability distribution."""
        import numpy as np
        prob_dist = np.array(prob_dist)
        prob_dist = prob_dist[prob_dist > 0]  # Remove zeros
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        return float(entropy)
