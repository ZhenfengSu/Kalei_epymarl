"""
K-2:4 Pattern-based Gumbel-Softmax Layer for Semi-structured Sparsity

This module implements the core 2:4 semi-structured pruning mechanism using:
1. Pattern-based projection (6 valid 2:4 sparse patterns)
2. Gumbel-Softmax for differentiable sampling
3. Dynamic heterogeneous scoring with EMA activation tracking

Corresponds to:
- Module A: Dynamic Heterogeneous Scoring
- Module B: Pattern-based Projection
- Training Dynamics: Temperature Annealing & Adaptive Resetting
"""

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class EMAActivationTracker:
    """
    Exponential Moving Average tracker for activation magnitudes.

    Solves the RL distribution shift problem by maintaining running statistics
    of input activations without requiring a separate calibration dataset.

    Attributes:
        momentum: EMA coefficient (default 0.99 for smooth tracking)
    """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.registered = False
        self.ema_activation = None

    def update(self, activation):
        """Update EMA statistics with new activation batch."""
        with th.no_grad():
            act_mag = th.abs(activation)
            if not self.registered:
                self.ema_activation = act_mag.mean(dim=0).clone()
                self.registered = True
            else:
                self.ema_activation = (
                    self.momentum * self.ema_activation
                    + (1 - self.momentum) * act_mag.mean(dim=0)
                )
        return self.ema_activation

    def reset(self, new_activation=None):
        """Reset EMA statistics (for adaptive resetting mechanism)."""
        if new_activation is not None:
            with th.no_grad():
                self.ema_activation = th.abs(new_activation).mean(dim=0).clone()
        else:
            self.registered = False
            self.ema_activation = None


class Pattern24Matrix:
    """
    2:4 Semi-structured Sparse Pattern Matrix.

    Defines the 6 valid sparse patterns for 1x4 weight groups:
    Pattern 0: [1, 1, 0, 0]  (indices 0,1 active)
    Pattern 1: [1, 0, 1, 0]  (indices 0,2 active)
    Pattern 2: [1, 0, 0, 1]  (indices 0,3 active)
    Pattern 3: [0, 1, 1, 0]  (indices 1,2 active)
    Pattern 4: [0, 1, 0, 1]  (indices 1,3 active)
    Pattern 5: [0, 0, 1, 1]  (indices 2,3 active)

    Shape: [6, 4] where each row is a valid 2:4 pattern.
    """

    def __init__(self, device):
        # Transposed pattern matrix for efficient computation
        # Shape: [4, 6] - each column is a pattern
        self.pattern_matrix = th.tensor([
            [1, 1, 1, 0, 0, 0],  # Weight position 0
            [1, 0, 0, 1, 1, 0],  # Weight position 1
            [0, 1, 0, 1, 0, 1],  # Weight position 2
            [0, 0, 1, 0, 1, 1],  # Weight position 3
        ], dtype=th.float32, device=device)

        # Original shape for mask reconstruction [6, 4]
        self.pattern_matrix_orig = self.pattern_matrix.T

    def project_to_patterns(self, scores):
        """
        Project importance scores to pattern logits.

        Args:
            scores: [N, 4] importance scores for each 1x4 weight group

        Returns:
            pattern_logits: [N, 6] logits for each of the 6 patterns

        Mathematical derivation:
            [a, b, c, d] @ M^T where M is [4, 6]
            = [a+b, a+c, a+d, b+c, b+d, c+d]
            Each element = sum of scores if we select that pattern
        """
        self.pattern_matrix = self.pattern_matrix.to(scores.device)
        return th.matmul(scores, self.pattern_matrix)

    def reconstruct_mask(self, pattern_probs):
        """
        Reconstruct binary masks from pattern probabilities.

        Args:
            pattern_probs: [N, 6] probability distribution over patterns

        Returns:
            masks: [N, 4] binary masks (2 active per 4 weights)

        Mathematical derivation:
            pi @ M where pi is [N, 6] and M is [6, 4]
        """
        self.pattern_matrix_orig = self.pattern_matrix_orig.to(pattern_probs.device)
        return th.matmul(pattern_probs, self.pattern_matrix_orig)


class SemiStructuredLinear24(nn.Linear):
    """
    Custom Linear layer with 2:4 semi-structured sparsity for each agent.

    Implements K-2:4 algorithm:
    1. Dynamic Heterogeneous Scoring: EMA activations * shared weights * sigmoid(alpha)
    2. Pattern-based Projection: Project to 6 valid 2:4 patterns
    3. Gumbel-Softmax Sampling: Differentiable discrete pattern selection
    4. Temperature Annealing: Gradually reduce sampling noise

    Args:
        n_agents: Number of heterogeneous agents
        hidden_dim: Hidden dimension for RNN
        temperature_init: Initial temperature for Gumbel-Softmax
        temperature_min: Minimum temperature (hard decision)
        ema_momentum: EMA coefficient for activation tracking
        hetero_init_scale: Initial scale for heterogeneity coefficients alpha
        *args, **kwargs: Arguments for nn.Linear
    """

    def __init__(
        self,
        n_agents,
        hidden_dim,
        temperature_init=5.0,
        temperature_min=0.1,
        ema_momentum=0.99,
        hetero_init_scale=0.01,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # Temperature annealing parameters
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.register_buffer('temperature', th.tensor(temperature_init))

        # Pattern matrix for 2:4 sparsity
        self.pattern_matrix = Pattern24Matrix(self.weight.device)

        # EMA activation tracker (one per layer, shared across agents)
        self.ema_tracker = EMAActivationTracker(momentum=ema_momentum)

        # Module A: Heterogeneity coefficients (agent-specific)
        # Each agent has its own alpha to modulate weight importance
        self.hetero_alpha = nn.Parameter(
            th.randn(n_agents, self.out_features, self.in_features) * hetero_init_scale
        )

        # For adaptive reset mechanism
        self.hetero_init_scale = hetero_init_scale

        # ========== Rewind & Finetune Mechanism ==========
        # Mask freezing for lottery ticket hypothesis style finetuning
        self.mask_frozen = False
        self.register_buffer('frozen_mask', None)  # Will store frozen masks per agent

    def _compute_heterogeneous_scores(self, agent_ids):
        """
        Module A: Dynamic Heterogeneous Scoring

        S_i = |W_shared| * EMA(|A_i|) * sigmoid(alpha_i)

        Returns:
            scores: [batch_size, out_features, in_features]
                Heterogeneous importance scores for each agent
        """
        batch_size = agent_ids.shape[0]

        # 1. Get EMA activation statistics [in_features]
        # Use running average if available, otherwise use ones
        if self.ema_tracker.ema_activation is not None:
            ema_act = self.ema_tracker.ema_activation  # [in_features]
        else:
            ema_act = th.ones(self.in_features, device=self.weight.device)

        # 2. Shared weight magnitude [out_features, in_features]
        w_mag = th.abs(self.weight)  # [out_features, in_features]

        # 3. Agent-specific heterogeneity coefficients
        # Expand for batch [batch_size, out_features, in_features]
        alpha = self.hetero_alpha[agent_ids]  # [batch_size, out_features, in_features]
        alpha_modulation = th.sigmoid(alpha)

        # 4. Compute final scores
        # w_mag: [out_features, in_features] -> [1, out_features, in_features]
        # ema_act: [in_features] -> [1, 1, in_features]
        # alpha_modulation: [batch_size, out_features, in_features]
        scores = (
            w_mag.unsqueeze(0) *
            ema_act.view(1, 1, -1) *
            alpha_modulation
        )  # [batch_size, out_features, in_features]

        return scores

    def _pattern_gumbel_softmax(self, scores):
        """
        Module B: Pattern-based Gumbel-Softmax Projection

        1. Project scores to pattern logits
        2. Apply Gumbel-Softmax for differentiable sampling
        3. Reconstruct binary masks from selected patterns

        Args:
            scores: [N, out_features*in_features] importance scores (flattened)

        Returns:
            masks: [N, out_features, in_features] binary 2:4 masks
            pattern_probs_soft: [N, n_groups, 6] soft probabilities (for diversity loss)
        """
        N = scores.shape[0]

        # Reshape to group into 1x4 units
        # Shape: [N, out_features, in_features]
        scores = scores.view(N, self.out_features, self.in_features)

        # Ensure divisible by 4 for 2:4 pattern
        if scores.shape[-1] % 4 != 0:
            # Pad if necessary
            pad_size = 4 - (scores.shape[-1] % 4)
            scores = F.pad(scores, (0, pad_size))
            padded = True
        else:
            padded = False

        # Reshape to [N, out_features, in_features//4, 4] for pattern grouping
        n_groups = scores.shape[2] // 4
        scores_grouped = scores.reshape(N, self.out_features, n_groups, 4)

        # Module B.1: Project to pattern logits
        # [N, out_features, n_groups, 4] @ [4, 6] -> [N, out_features, n_groups, 6]
        pattern_logits = self.pattern_matrix.project_to_patterns(
            scores_grouped.reshape(-1, 4)
        ).reshape(N, self.out_features, n_groups, 6)
        
        # ==========================================
        # === 核心修改：在这里添加 LayerNorm ===
        # ==========================================
        # 对最后一个维度 (6个模式的评分) 进行归一化
        # 这确保了无论权重的绝对大小如何，这6个模式都在同一起跑线上竞争
        pattern_logits = F.layer_norm(pattern_logits, (6,))
        
        # Module B.2: Gumbel-Softmax sampling
        # Forward: one-hot (discrete), Backward: soft (continuous)
        pattern_probs_hard = F.gumbel_softmax(
            pattern_logits, tau=self.temperature, hard=True, dim=-1
        )
        pattern_probs_soft = F.softmax(pattern_logits / self.temperature, dim=-1)

        # Module B.3: Reconstruct masks from patterns
        # [N, out_features, n_groups, 6] @ [6, 4] -> [N, out_features, n_groups, 4]
        masks_grouped = self.pattern_matrix.reconstruct_mask(pattern_probs_hard)

        # Reshape back to [N, out_features, in_features]
        masks = masks_grouped.reshape(N, self.out_features, -1)

        # Remove padding if added
        if padded:
            masks = masks[:, :, :self.in_features]

        return masks, pattern_probs_soft

    def forward(self, x, agent_ids):
        """
        Forward pass with 2:4 semi-structured sparsity.

        Args:
            x: [batch_size, in_features] input activations
            agent_ids: [batch_size] agent indices for heterogeneous masks

        Returns:
            output: [batch_size, out_features] layer output
        """
        # Update EMA statistics (Module A) - only if mask not frozen
        if not self.mask_frozen:
            self.ema_tracker.update(x)
            # Compute heterogeneous scores (Module A)
            scores = self._compute_heterogeneous_scores(agent_ids)
            # Apply pattern-based Gumbel-Softmax (Module B)
            masks, pattern_probs = self._pattern_gumbel_softmax(scores)
            # Store pattern probabilities for diversity loss (Module C)
            self.last_pattern_probs = pattern_probs.detach()
        else:
            # Use frozen mask during finetuning
            masks = self.frozen_mask[agent_ids]  # [batch_size, out_features, in_features]
            # Set pattern probs to None to indicate diversity loss should be skipped
            self.last_pattern_probs = None

        # Apply masks to shared weights
        # masks: [batch_size, out_features, in_features]
        # self.weight: [out_features, in_features]
        w_masked = self.weight.unsqueeze(0) * masks  # [batch_size, out_features, in_features]

        # Standard linear transformation
        # [batch_size, out_features, in_features] @ [batch_size, in_features, 1]
        result = th.bmm(w_masked, x.unsqueeze(-1)).squeeze(-1) + self.bias.unsqueeze(0)

        return result

    def set_temperature(self, temp):
        """Update temperature for annealing schedule."""
        self.temperature.fill_(temp)

    def get_pattern_probs(self):
        """Get last computed pattern probabilities (for diversity loss)."""
        return getattr(self, 'last_pattern_probs', None)

    def anneal_temperature(self, progress, start_step=0, end_step=1000):
        """
        Linear temperature annealing from temperature_init to temperature_min.

        Args:
            progress: Current training progress (0.0 to 1.0)
            start_step: Step to start annealing
            end_step: Step to end annealing
        """
        if progress < start_step / end_step:
            temp = self.temperature_init
        elif progress > 1.0:
            temp = self.temperature_min
        else:
            # Linear annealing
            temp = self.temperature_init - (self.temperature_init - self.temperature_min) * progress

        self.set_temperature(temp)

    def reset_hetero_alpha(self, reset_mask=None):
        """
        Reset heterogeneity coefficients (adaptive resetting mechanism).

        Args:
            reset_mask: [n_agents, out_features, in_features] boolean mask
                If None, randomly reset 10% of coefficients
        """
        with th.no_grad():
            # 1. 统一构建布尔类型的 mask
            if reset_mask is None:
                # 情况 A: 默认 10% 概率
                mask = th.rand_like(self.hetero_alpha) < 0.1
            elif isinstance(reset_mask, float):
                # 情况 B: 传入的是浮点数概率 (例如 0.1)
                mask = th.rand_like(self.hetero_alpha) < reset_mask
            else:
                # 情况 C: 传入的是 Tensor Mask，直接使用
                mask = reset_mask

            # 确保 mask 和 alpha 在同一个设备上
            if isinstance(mask, th.Tensor):
                mask = mask.to(self.hetero_alpha.device)

            # 2. 生成新的随机初始值
            new_alpha = th.randn_like(self.hetero_alpha) * self.hetero_init_scale
            self.hetero_alpha.copy_(
                th.where(mask, new_alpha, self.hetero_alpha)
            )

    def reset_ema(self, new_activation=None):
        """Reset EMA statistics."""
        self.ema_tracker.reset(new_activation)

    def get_sparsity(self):
        """Get actual sparsity (should be close to 50% for 2:4)."""
        with th.no_grad():
            # Sample from all agents
            agent_ids = th.arange(self.n_agents, device=self.weight.device)
            scores = self._compute_heterogeneous_scores(agent_ids)
            masks, _ = self._pattern_gumbel_softmax(scores)
            sparsity = 1.0 - masks.mean().item()
        return sparsity

    def get_pattern_distribution(self):
        """Get distribution over patterns for analysis."""
        pattern_probs = self.get_pattern_probs()
        if pattern_probs is not None:
            # Average over batch and features
            return pattern_probs.mean(dim=(0, 1)).mean(dim=0)  # [6]
        return None

    # ========== Rewind & Finetune Methods ==========

    def freeze_mask(self):
        """
        Freeze the current mask pattern for all agents.

        This implements the lottery ticket hypothesis "rewind" step:
        - Samples the current mask for all agents
        - Stores the mask in frozen_mask buffer
        - Sets mask_frozen flag to True

        During finetuning:
        - Gumbel-Softmax is disabled
        - EMA update is disabled
        - The frozen mask is used directly
        - Only weights W are optimized, not heterogeneity coefficients alpha
        """
        with th.no_grad():
            # Get agent IDs for all agents
            agent_ids = th.arange(self.n_agents, device=self.weight.device)

            # Compute scores for all agents
            scores = self._compute_heterogeneous_scores(agent_ids)

            # Get current masks (using Gumbel-Softmax)
            masks, _ = self._pattern_gumbel_softmax(scores)

            # Store masks per agent [n_agents, out_features, in_features]
            self.frozen_mask = masks.clone()

            # Set frozen flag
            self.mask_frozen = True

    def unfreeze_mask(self):
        """
        Unfreeze the mask (restore normal training).

        This allows resuming mask exploration if needed.
        """
        self.mask_frozen = False
        self.frozen_mask = None

    def is_mask_frozen(self):
        """Check if mask is currently frozen."""
        return self.mask_frozen


def create_k24_linear(n_agents, hidden_dim, **kwargs):
    """Factory function to create SemiStructuredLinear24 layer."""
    return partial(SemiStructuredLinear24, n_agents=n_agents, hidden_dim=hidden_dim, **kwargs)
