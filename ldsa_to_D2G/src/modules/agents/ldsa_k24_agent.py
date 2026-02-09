"""
LDSA Agent with TAAE-style Attention and K24 2:4 Sparsity.

This module extends the LDSA agent with:
1. TAAE-style attention mechanism integrating role, global state, and observations
2. K24 2:4 semi-structured sparsity for efficient inference
3. Global state usage in first-stage (centralized) training

Key components:
- OptimizedDynamicSubtaskGenerator: Dynamic subtask count prediction
- RoleStateAttention: Multi-head attention for role-state-obs integration
- K24SparseLDSAAgent: Main agent with sparse layers and attention

References:
- TAAE: /mnt/lc_gpu_test/marl_final_exp/ptde_ldsa_attn
- K24: /mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/Kaleidoscope/Kalei_SMACv2
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from modules.attention.role_state_attention import MultiAgentAttentionModule
from modules.layer.k24_sparse_linear import SemiStructuredLinear24


class OptimizedDynamicSubtaskGenerator(nn.Module):
    """
    Optimized version with global state support and optional K24 sparsity.

    Enhanced from original ldsa_agent_optim.py to:
    1. Support global state as additional context
    2. Optional K24 sparse linear layers
    3. More efficient computation

    Args:
        args: Configuration arguments
        embed_dim: Embedding dimension
        use_k24: Whether to use K24 sparse layers
    """

    def __init__(self, args, embed_dim, use_k24=False):
        super(OptimizedDynamicSubtaskGenerator, self).__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.state_dims = args.state_dims
        self.use_k24 = use_k24
        self.n_agents = getattr(args, 'n_agents', 2)

        self.max_subtasks = getattr(args, 'max_subtasks', 5)
        self.min_subtasks = getattr(args, 'min_subtasks', 2)
        self.num_subtask_options = self.max_subtasks - self.min_subtasks + 1

        # Network structure
        if use_k24 and hasattr(args, 'K24_use_global_state') and args.K24_use_global_state:
            # Use K24 sparse layer with global state
            input_dim = embed_dim + args.state_dims
            self.context_fusion = SemiStructuredLinear24(
                n_agents=self.n_agents,
                hidden_dim=embed_dim,
                in_features=input_dim,
                out_features=embed_dim,
                temperature_init=getattr(args, 'K24_temperature_init', 5.0),
                temperature_min=getattr(args, 'K24_temperature_min', 0.1),
            )
        else:
            # Standard linear layer
            self.context_fusion = nn.Linear(embed_dim + args.state_dims, embed_dim)

        # Subtask count predictor
        self.subtask_count_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.num_subtask_options)
        )

        self.temperature = getattr(args, 'gumbel_temperature', 1.0)

        # Pre-computed indices
        self.register_buffer('subtask_indices',
                           torch.arange(self.min_subtasks, self.max_subtasks + 1, dtype=torch.float32))

        # Caching mechanism
        self.cache_enabled = getattr(args, 'cache_subtask_count', True)
        self.cache_steps = getattr(args, 'cache_steps', 10)
        self.step_counter = 0
        self.cached_n_subtasks = self.max_subtasks

        # Agent IDs for K24 (all agents use same generator, so use single ID)
        self.register_buffer('k24_agent_id', torch.tensor([0]))

    def forward(self, agent_features, global_context=None, training=True):
        """
        Forward pass to predict number of subtasks.

        Args:
            agent_features: [batch_size, n_agents, embed_dim] agent features
            global_context: [batch_size, state_dims] global state
            training: Whether in training mode

        Returns:
            n_subtasks: Number of subtasks to use
        """
        # Caching for inference
        if self.cache_enabled and not training:
            self.step_counter += 1
            if self.step_counter % self.cache_steps != 0:
                return self.cached_n_subtasks

        batch_size = agent_features.shape[0]

        # Aggregate agent features
        aggregated_features = agent_features.sum(dim=1) / agent_features.shape[1]

        # Fuse with global context
        if global_context is not None:
            context_features = torch.cat([aggregated_features, global_context], dim=-1)
            if self.use_k24:
                fused_features = self.context_fusion(context_features, self.k24_agent_id.expand(batch_size))
            else:
                fused_features = self.context_fusion(context_features)
        else:
            fused_features = aggregated_features

        # Predict logits
        subtask_count_logits = self.subtask_count_predictor(fused_features)

        # Sampling strategy
        if training:
            subtask_count_probs = F.gumbel_softmax(
                subtask_count_logits,
                tau=max(0.1, self.temperature * 0.5),
                hard=True,
                dim=-1
            )
        else:
            max_indices = torch.argmax(subtask_count_logits, dim=-1)
            subtask_count_probs = torch.zeros_like(subtask_count_logits)
            subtask_count_probs.scatter_(1, max_indices.unsqueeze(1), 1.0)

        # Compute number of subtasks
        subtask_indices_expanded = self.subtask_indices.unsqueeze(0).expand(batch_size, -1)
        expected_n_subtasks = (subtask_count_probs * subtask_indices_expanded).sum(dim=-1)

        n_subtasks = int(expected_n_subtasks.max().item())
        n_subtasks = max(self.min_subtasks, min(n_subtasks, self.max_subtasks))

        # Update cache
        if self.cache_enabled:
            self.cached_n_subtasks = n_subtasks

        return n_subtasks


class K24SparseLDSAAgent(nn.Module):
    """
    LDSA Agent with TAAE-style Attention and K24 2:4 Sparsity.

    This agent integrates:
    1. Dynamic subtask generation with global state
    2. Role-state-observation attention mechanism
    3. K24 sparse layers for efficient inference
    4. Optimized computation for both centralized and decentralized execution

    Args:
        input_shape: Dimension of input observations
        args: Configuration arguments
    """

    def __init__(self, input_shape, args):
        super(K24SparseLDSAAgent, self).__init__()
        self.args = args

        # Check if using K24 sparsity
        self.use_k24 = getattr(args, 'use_k24', False)
        self.use_attention = getattr(args, 'use_attention', True)
        self.use_global_state = getattr(args, 'K24_use_global_state', True)

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        # K24 parameters
        if self.use_k24:
            self.temperature_init = getattr(args, 'K24_temperature_init', 5.0)
            self.temperature_min = getattr(args, 'K24_temperature_min', 0.1)
            self.anneal_end_step = getattr(args, 'K24_anneal_end_step', 100000)
            self.K24_linear = lambda in_features, out_features: SemiStructuredLinear24(
                n_agents=self.n_agents,
                hidden_dim=args.rnn_hidden_dim,
                in_features=in_features,
                out_features=out_features,
                temperature_init=self.temperature_init,
                temperature_min=self.temperature_min,
            )
        else:
            self.K24_linear = nn.Linear

        # Agent embedding network
        self.fc1_agent_embed = self.K24_linear(input_shape, args.rnn_hidden_dim)
        self.rnn_agent_embed = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_agent_embed = self.K24_linear(args.rnn_hidden_dim, args.agent_subtask_embed_dim)

        # Subtask representation
        if args.subtask_repr_layers == 2:
            self.subtask_embed_net = nn.Sequential(
                nn.Linear(args.max_subtasks, args.agent_subtask_embed_dim),
                nn.ReLU(),
                nn.Linear(args.agent_subtask_embed_dim, args.agent_subtask_embed_dim)
            )
        elif args.subtask_repr_layers == 1:
            self.subtask_embed_net = nn.Linear(args.max_subtasks, args.agent_subtask_embed_dim, bias=False)

        # Subtask policy network
        self.fc1_subtask_policy = self.K24_linear(input_shape, args.rnn_hidden_dim)
        self.rnn_subtask_policy = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        if not args.subtask_policy_use_hypernet:
            self.fc2_subtask_policy = self.K24_linear(args.rnn_hidden_dim, args.max_subtasks * args.n_actions)
        else:
            self.fc2_w = self.K24_linear(args.agent_subtask_embed_dim, args.rnn_hidden_dim * args.n_actions)
            self.fc2_b = self.K24_linear(args.agent_subtask_embed_dim, args.n_actions)

        # Optimized dynamic subtask generator with K24 option
        self.dynamic_subtask_generator = OptimizedDynamicSubtaskGenerator(
            args, args.agent_subtask_embed_dim, use_k24=self.use_k24 and self.use_global_state
        )

        # Attention module (if enabled)
        if self.use_attention and self.use_global_state:
            self.attention_module = MultiAgentAttentionModule(
                state_dims=args.state_dims,
                obs_dims=input_shape,
                role_dim=args.agent_subtask_embed_dim,
                z_dims=getattr(args, 'attention_z_dims', 64),
                num_heads=getattr(args, 'attention_num_heads', 4),
                hidden_dim=getattr(args, 'attention_hidden_dim', 64),
                use_global_state=self.use_global_state
            )

        # Pre-computed buffers
        self.register_buffer('mask_template', torch.zeros(1, 1, args.max_subtasks))
        self.register_buffer('subtask_one_hot',
                           torch.eye(args.max_subtasks).unsqueeze(0))

    def init_hidden_subtask_policy(self):
        return self.fc1_subtask_policy.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_hidden_agent_embed(self):
        return self.fc1_agent_embed.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def _get_agent_ids(self, batch_size):
        """Get agent IDs for K24 sparse layers."""
        if self.use_k24:
            # Create agent IDs: [0, 1, ..., n_agents-1, 0, 1, ..., n_agents-1, ...]
            # Repeat the pattern to cover the entire batch
            n_repeats = (batch_size + self.n_agents - 1) // self.n_agents  # Ceiling division
            agent_ids = torch.arange(self.n_agents, device=self.fc1_agent_embed.weight.device)
            agent_ids = agent_ids.unsqueeze(0).repeat(n_repeats, 1).flatten()[:batch_size]
            return agent_ids
        return None

    def forward(self, inputs, hidden_state_subtask_policy, hidden_state_agent_embed, state, test_mode=False):
        """
        Forward pass with attention and K24 sparsity.

        Args:
            inputs: [batch_size * n_agents, input_shape] local observations
            hidden_state_subtask_policy: Hidden state for subtask policy RNN
            hidden_state_agent_embed: Hidden state for agent embedding RNN
            state: [batch_size, state_dims] global state (used in centralized training)
            test_mode: Whether in test mode

        Returns:
            q: Q-values
            h_subtask_policy: Updated subtask policy hidden state
            h_agent_embed: Updated agent embedding hidden state
            subtask_prob_logit: Subtask selection logits
            subtask_embed: Subtask embeddings
        """
        
        n_agents = self.args.n_agents
        batch_size = inputs.shape[0]
        # Reshape inputs for agent-wise processing
        inputs_flat = inputs.reshape(-1, inputs.shape[-1])  # [batch * n_agents, input_shape]

        # Get agent IDs for K24
        agent_ids = self._get_agent_ids(inputs_flat.shape[0])

        # Agent embedding
        if self.use_k24:
            x_agent_embed = F.relu(self.fc1_agent_embed(inputs_flat, agent_ids))
        else:
            x_agent_embed = F.relu(self.fc1_agent_embed(inputs_flat))

        h_in_agent_embed = hidden_state_agent_embed.reshape(-1, self.args.rnn_hidden_dim)
        h_agent_embed = self.rnn_agent_embed(x_agent_embed, h_in_agent_embed)

        if self.use_k24:
            agent_embed = self.fc2_agent_embed(h_agent_embed, agent_ids)
        else:
            agent_embed = self.fc2_agent_embed(h_agent_embed)

        agent_embed = agent_embed.reshape(-1, n_agents, self.args.agent_subtask_embed_dim)

        # Dynamic subtask generation with global state
        if self.training or (hasattr(self, '_step_count') and self._step_count % 5 == 0):
            n_subtasks = self.dynamic_subtask_generator(agent_embed, state, self.training)
            self._cached_n_subtasks = n_subtasks
        else:
            n_subtasks = getattr(self, '_cached_n_subtasks', self.args.max_subtasks)

        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1

        # Subtask representation
        bs = agent_embed.shape[0]
        subtask_one_hot = self.subtask_one_hot.expand(bs, -1, -1)
        subtask_embed = self.subtask_embed_net(subtask_one_hot)
        if self.args.use_tanh:
            subtask_embed = F.tanh(subtask_embed)

        # Apply attention to enhance agent embeddings
        if self.use_attention and self.use_global_state and state is not None:
            # Enhance agent embeddings with attention
            enhanced_embed, attn_weights = self.attention_module(
                role_emb=agent_embed,
                obs=inputs_flat.reshape(bs, n_agents, -1),
                state=state,
                agent_mask=None
            )
            # Use enhanced embeddings for subtask selection
            agent_embed_for_selection = enhanced_embed
        else:
            agent_embed_for_selection = agent_embed
            attn_weights = None

        # Subtask policy
        if self.use_k24:
            x_subtask_policy = F.relu(self.fc1_subtask_policy(inputs_flat, agent_ids))
        else:
            x_subtask_policy = F.relu(self.fc1_subtask_policy(inputs_flat))

        h_in_subtask_policy = hidden_state_subtask_policy.reshape(-1, self.args.rnn_hidden_dim)
        h_subtask_policy = self.rnn_subtask_policy(x_subtask_policy, h_in_subtask_policy)

        if not self.args.subtask_policy_use_hypernet:
            if self.use_k24:
                q = self.fc2_subtask_policy(h_subtask_policy, agent_ids)
            else:
                q = self.fc2_subtask_policy(h_subtask_policy)
            q = q.reshape(-1, self.args.max_subtasks, self.args.n_actions)
        else:
            subtask_embed_detach = subtask_embed.clone().detach()[0]
            if self.use_k24:
                w2 = self.fc2_w(subtask_embed_detach, agent_ids[:subtask_embed_detach.shape[0]])
                b2 = self.fc2_b(subtask_embed_detach, agent_ids[:subtask_embed_detach.shape[0]])
            else:
                w2 = self.fc2_w(subtask_embed_detach)
                b2 = self.fc2_b(subtask_embed_detach)
            w2 = w2.unsqueeze(0).expand(bs * n_agents, -1, -1).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
            b2 = b2.unsqueeze(0).expand(bs * n_agents, -1, -1).reshape(-1, 1, self.args.n_actions)
            h_subtask_policy_ = h_subtask_policy.unsqueeze(1).expand(-1, self.args.max_subtasks, -1).reshape(-1, 1, self.args.rnn_hidden_dim)
            q = th.bmm(h_subtask_policy_, w2) + b2
            q = q.reshape(-1, self.args.max_subtasks, self.args.n_actions)

        # Subtask selection with enhanced embeddings
        subtask_prob_logit = th.bmm(agent_embed_for_selection, subtask_embed.permute(0, 2, 1))

        # Mask for dynamic subtask count
        if n_subtasks < self.args.max_subtasks:
            mask = self.mask_template.expand_as(subtask_prob_logit).clone()
            mask[..., n_subtasks:] = -float('inf')
            subtask_prob_logit = subtask_prob_logit + mask

        if self.args.random_sele:
            subtask_prob_logit = th.rand_like(subtask_prob_logit)

        if test_mode and self.args.test_argmax:
            prob_max = th.max(subtask_prob_logit, dim=-1, keepdim=True)[1]
            subtask_prob = th.zeros_like(subtask_prob_logit).scatter_(-1, prob_max, 1)
        else:
            if self.args.sft_way == "softmax":
                subtask_prob = F.softmax(subtask_prob_logit, dim=-1)
            elif self.args.sft_way == "gumbel_softmax":
                subtask_prob = F.gumbel_softmax(subtask_prob_logit, hard=True, dim=-1)

        subtask_prob = subtask_prob.reshape(-1, 1, self.args.max_subtasks)

        if self.args.evaluate:
            print('chosen_subtask_prob', subtask_prob.reshape(self.args.n_agents, self.args.max_subtasks))

        q = th.bmm(subtask_prob, q).squeeze(1)

        return q, h_subtask_policy, h_agent_embed, subtask_prob_logit, subtask_embed

    def get_k24_layers(self):
        """Get all K24 sparse layers for temperature annealing."""
        if not self.use_k24:
            return []

        layers = []
        # Agent embedding layers
        if hasattr(self, 'fc1_agent_embed') and isinstance(self.fc1_agent_embed, SemiStructuredLinear24):
            layers.append(self.fc1_agent_embed)
        if hasattr(self, 'fc2_agent_embed') and isinstance(self.fc2_agent_embed, SemiStructuredLinear24):
            layers.append(self.fc2_agent_embed)

        # Subtask policy layers
        if hasattr(self, 'fc1_subtask_policy') and isinstance(self.fc1_subtask_policy, SemiStructuredLinear24):
            layers.append(self.fc1_subtask_policy)

        # Either fc2_subtask_policy or hypernetwork (fc2_w, fc2_b)
        if hasattr(self, 'fc2_subtask_policy') and isinstance(self.fc2_subtask_policy, SemiStructuredLinear24):
            layers.append(self.fc2_subtask_policy)
        if hasattr(self, 'fc2_w') and isinstance(self.fc2_w, SemiStructuredLinear24):
            layers.append(self.fc2_w)
        if hasattr(self, 'fc2_b') and isinstance(self.fc2_b, SemiStructuredLinear24):
            layers.append(self.fc2_b)

        return layers

    def freeze_masks(self):
        """Freeze all K24 sparse masks."""
        for layer in self.get_k24_layers():
            layer.freeze_mask()

    def unfreeze_masks(self):
        """Unfreeze all K24 sparse masks."""
        for layer in self.get_k24_layers():
            layer.unfreeze_mask()

    def set_temperature(self, temp):
        """Set temperature for all K24 layers."""
        for layer in self.get_k24_layers():
            layer.set_temperature(temp)

    def get_pattern_probs(self):
        """Get pattern probabilities from all K24 layers."""
        probs = {}
        for i, layer in enumerate(self.get_k24_layers()):
            probs[f'layer_{i}'] = layer.get_pattern_probs()
        return probs
