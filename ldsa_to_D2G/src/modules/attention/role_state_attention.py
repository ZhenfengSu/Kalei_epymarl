"""
Role-State-Observation Attention Module for LDSA.

This module implements the attention mechanism that integrates:
1. Role embeddings (from subtask selection probabilities)
2. Global state (environment-provided, used in centralized training)
3. Local observations

This follows the TAAE (Team-Aware Attention Extraction) approach but adapted
for LDSA's dynamic subtask generation.

Architecture:
    - Query: Derived from role/subtask embeddings
    - Key/Value: Derived from global state + observation fusion
    - Multi-head attention for information extraction
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


class GlobalStateFusion(nn.Module):
    """
    Fuses global state with local observations using hypernetwork approach.

    In TAAE, this is called the "Coach Network" (GIN module).
    During centralized training, we use ground-truth global state.
    During decentralized execution, this can be replaced with a learned generator.

    Args:
        state_dims: Dimension of global state
        obs_dims: Dimension of local observations
        z_dims: Latent dimension for fusion
        hidden_dims: Hidden layer dimensions
    """

    def __init__(self, state_dims, obs_dims, z_dims=64, hidden_dims=128):
        super(GlobalStateFusion, self).__init__()
        self.state_dims = state_dims
        self.obs_dims = obs_dims
        self.z_dims = z_dims

        # Hypernetwork: observation -> weights for processing global state
        # This allows personalized state processing per agent
        self.w1_net = nn.Sequential(
            nn.Linear(obs_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, state_dims * z_dims)
        )

        self.b1_net = nn.Sequential(
            nn.Linear(obs_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, z_dims)
        )

        # Direct observation encoding
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, z_dims)
        )

    def forward(self, state, obs):
        """
        Fuse global state and local observation.

        Args:
            state: [batch_size, state_dims] global state
            obs: [batch_size, obs_dims] local observations

        Returns:
            fused: [batch_size, z_dims] fused representation
        """
        batch_size = state.shape[0]

        # Hypernetwork approach: use obs to generate weights for state processing
        w1 = self.w1_net(obs).view(batch_size, self.state_dims, self.z_dims)
        b1 = self.b1_net(obs).view(batch_size, 1, self.z_dims)

        # Process global state with obs-conditioned weights
        # [batch, state, z] = [batch, state, z] matmul [batch, state, 1]
        z_state = th.bmm(w1.transpose(1, 2), state.unsqueeze(-1)).squeeze(-1) + b1.squeeze(1)

        # Encode observation directly
        z_obs = self.obs_encoder(obs)

        # Fuse state and observation representations
        z_fused = z_state + z_obs  # Residual connection

        return z_fused


class RoleStateAttention(nn.Module):
    """
    Multi-Head Attention for Role-State-Observation integration.

    This module uses subtask probabilities (role assignments) as queries to
    extract relevant information from the global state-observation fusion.

    Architecture:
        1. Role Query: From subtask probability distribution
        2. State-Obs Key/Value: From GlobalStateFusion output
        3. Multi-Head Attention: Compute weighted combination
        4. Output Projection: Produce context vector for decision making

    Args:
        role_dim: Dimension of role/subtask embeddings
        z_dims: Dimension of fused state-observation representation
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for projections
    """

    def __init__(self, role_dim, z_dims=64, num_heads=4, hidden_dim=64):
        super(RoleStateAttention, self).__init__()
        self.role_dim = role_dim
        self.z_dims = z_dims
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query projection: from role/subtask probabilities
        self.q_proj = nn.Linear(role_dim, hidden_dim)

        # Key/Value projections: from state-obs fusion
        self.k_proj = nn.Linear(z_dims, hidden_dim)
        self.v_proj = nn.Linear(z_dims, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, role_dim)

        # Scale factor for scaled dot-product attention
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, role_emb, state_obs_fused, mask=None):
        """
        Apply role-state attention.

        Args:
            role_emb: [batch_size, role_dim] role/subtask embedding
            state_obs_fused: [batch_size, n_agents, z_dims] fused state-obs for all agents
            mask: Optional attention mask [batch_size, n_agents]

        Returns:
            context: [batch_size, role_dim] attention context vector
            attn_weights: [batch_size, num_heads, n_agents] attention weights (for analysis)
        """
        batch_size, n_agents, _ = state_obs_fused.shape

        # Project role to query
        # [batch, hidden]
        Q = self.q_proj(role_emb)  # [batch, hidden_dim]

        # Reshape for multi-head: [batch, num_heads, head_dim]
        Q = Q.view(batch_size, self.num_heads, self.head_dim)

        # Project state-obs to key/value
        # [batch, n_agents, hidden]
        K = self.k_proj(state_obs_fused)
        V = self.v_proj(state_obs_fused)

        # Reshape for multi-head: [batch, n_agents, num_heads, head_dim]
        K = K.view(batch_size, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V shape: [batch, num_heads, n_agents, head_dim]

        # Scaled dot-product attention
        # [batch, num_heads, 1, n_agents]
        attn_scores = th.matmul(Q.unsqueeze(2), K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, n_agents]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax over agents
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, num_heads, 1, n_agents]

        # Apply attention to values
        # [batch, num_heads, 1, head_dim]
        context = th.matmul(attn_weights, V).squeeze(2)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim)

        # Output projection
        context = self.out_proj(context)  # [batch, role_dim]

        return context, attn_weights.squeeze(2)  # [batch, role_dim], [batch, num_heads, n_agents]


class MultiAgentAttentionModule(nn.Module):
    """
    Complete attention module for multi-agent coordination.

    This module combines:
    1. Global state fusion (state + obs)
    2. Role-based attention for each agent
    3. Multi-head processing

    Args:
        state_dims: Dimension of global state
        obs_dims: Dimension of local observations
        role_dim: Dimension of role/subtask embeddings
        z_dims: Latent dimension for fusion
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension
        use_global_state: Whether to use global state (True for centralized training)
    """

    def __init__(self, state_dims, obs_dims, role_dim, z_dims=64, num_heads=4, hidden_dim=64, use_global_state=True):
        super(MultiAgentAttentionModule, self).__init__()
        self.use_global_state = use_global_state

        if use_global_state:
            # Global state fusion module
            self.state_fusion = GlobalStateFusion(state_dims, obs_dims, z_dims)
            attn_input_dim = z_dims
        else:
            # Without global state, use only observations
            attn_input_dim = obs_dims

        # Attention module
        self.attention = RoleStateAttention(role_dim, attn_input_dim, num_heads, hidden_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(role_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(role_dim, role_dim * 4),
            nn.ReLU(),
            nn.Linear(role_dim * 4, role_dim)
        )

        self.layer_norm2 = nn.LayerNorm(role_dim)

    def forward(self, role_emb, obs, state=None, agent_mask=None):
        """
        Apply complete attention processing.

        Args:
            role_emb: [batch_size, n_agents, role_dim] role embeddings for all agents
            obs: [batch_size, n_agents, obs_dims] local observations
            state: [batch_size, state_dims] global state (only if use_global_state=True)
            agent_mask: [batch_size, n_agents] optional mask for active agents

        Returns:
            enhanced_role: [batch_size, n_agents, role_dim] enhanced role embeddings
            attn_weights: [batch_size, n_agents, num_heads, n_agents] attention weights
        """
        batch_size, n_agents, role_dim = role_emb.shape

        if self.use_global_state:
            if state is None:
                raise ValueError("Global state must be provided when use_global_state=True")

            # Process each agent separately
            enhanced_roles = []
            all_attn_weights = []

            for i in range(n_agents):
                # Fuse state and obs for agent i
                state_obs_fused = self.state_fusion(state, obs[:, i, :])  # [batch, z_dims]

                # Expand to include all agents for cross-attention
                # For now, we use the same fused representation for all agents
                # In a more sophisticated version, we could create different fusions per agent
                state_obs_fused_expanded = state_obs_fused.unsqueeze(1).expand(-1, n_agents, -1)

                # Apply attention
                context, attn_w = self.attention(
                    role_emb[:, i, :],
                    state_obs_fused_expanded,
                    agent_mask
                )

                # Residual connection + layer norm
                enhanced = self.layer_norm(role_emb[:, i, :] + context)

                # Feed-forward network + residual + layer norm
                enhanced = self.layer_norm2(enhanced + self.ffn(enhanced))

                enhanced_roles.append(enhanced)
                all_attn_weights.append(attn_w)

            enhanced_role = th.stack(enhanced_roles, dim=1)  # [batch, n_agents, role_dim]
            attn_weights = th.stack(all_attn_weights, dim=1)  # [batch, n_agents, num_heads, n_agents]
        else:
            # Without global state, apply self-attention on role embeddings
            # This is a fallback for decentralized-only execution
            enhanced_role = role_emb
            attn_weights = None

        return enhanced_role, attn_weights
