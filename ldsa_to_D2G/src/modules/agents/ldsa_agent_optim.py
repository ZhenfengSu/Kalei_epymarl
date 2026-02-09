import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

class OptimizedDynamicSubtaskGenerator(nn.Module):
    def __init__(self, args, embed_dim):
        super(OptimizedDynamicSubtaskGenerator, self).__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.state_dims = args.state_dims
        
        self.max_subtasks = getattr(args, 'max_subtasks', 5)
        self.min_subtasks = getattr(args, 'min_subtasks', 2)
        self.num_subtask_options = self.max_subtasks - self.min_subtasks + 1
        
        # 简化网络结构，减少计算量
        self.context_fusion = nn.Linear(embed_dim + args.state_dims, embed_dim)
        
        # 更轻量的预测器
        self.subtask_count_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),  # 减少中间层大小
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.num_subtask_options)
        )
        
        self.temperature = getattr(args, 'gumbel_temperature', 1.0)
        
        # 预计算的索引，避免重复创建
        self.register_buffer('subtask_indices', 
                           torch.arange(self.min_subtasks, self.max_subtasks + 1, dtype=torch.float32))
        
        # 缓存机制
        self.cache_enabled = getattr(args, 'cache_subtask_count', True)
        self.cache_steps = getattr(args, 'cache_steps', 10)  # 每10步更新一次
        self.step_counter = 0
        self.cached_n_subtasks = self.max_subtasks  # 默认值
        
    def forward(self, agent_features, global_context=None, training=True):
        # 缓存机制：不是每步都重新计算
        if self.cache_enabled and not training:
            self.step_counter += 1
            if self.step_counter % self.cache_steps != 0:
                return self.cached_n_subtasks
        
        batch_size = agent_features.shape[0]
        
        # 1. 快速聚合（避免mean操作的开销）
        aggregated_features = agent_features.sum(dim=1) / agent_features.shape[1]
        
        # 2. 简化的融合
        if global_context is not None:
            context_features = torch.cat([aggregated_features, global_context], dim=-1)
            fused_features = self.context_fusion(context_features)
        else:
            fused_features = aggregated_features
        
        # 3. 预测logits
        subtask_count_logits = self.subtask_count_predictor(fused_features)
        
        # 4. 简化的采样策略
        if training:
            # 训练时使用更低的温度，减少计算复杂度
            subtask_count_probs = F.gumbel_softmax(
                subtask_count_logits, 
                tau=max(0.1, self.temperature * 0.5),  # 降低温度
                hard=True,
                dim=-1
            )
        else:
            # 测试时直接argmax，避免gumbel_softmax
            max_indices = torch.argmax(subtask_count_logits, dim=-1)
            subtask_count_probs = torch.zeros_like(subtask_count_logits)
            subtask_count_probs.scatter_(1, max_indices.unsqueeze(1), 1.0)
        
        # 5. 优化的数量计算
        # 使用预计算的索引
        subtask_indices_expanded = self.subtask_indices.unsqueeze(0).expand(batch_size, -1)
        expected_n_subtasks = (subtask_count_probs * subtask_indices_expanded).sum(dim=-1)
        
        # 简化的整数转换
        n_subtasks = int(expected_n_subtasks.max().item())
        n_subtasks = max(self.min_subtasks, min(n_subtasks, self.max_subtasks))
        
        # 更新缓存
        if self.cache_enabled:
            self.cached_n_subtasks = n_subtasks
        
        return n_subtasks

class LDSAAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LDSAAgent, self).__init__()
        self.args = args

        # 原有网络保持不变
        self.fc1_agent_embed = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_agent_embed = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_agent_embed = nn.Linear(args.rnn_hidden_dim, args.agent_subtask_embed_dim)

        if args.subtask_repr_layers == 2:
            self.subtask_embed_net = nn.Sequential(
                nn.Linear(args.max_subtasks, args.agent_subtask_embed_dim),
                nn.ReLU(),
                nn.Linear(args.agent_subtask_embed_dim, args.agent_subtask_embed_dim)
            )
        elif args.subtask_repr_layers == 1:
            self.subtask_embed_net = nn.Linear(args.max_subtasks, args.agent_subtask_embed_dim, bias=False)
        
        self.fc1_subtask_policy = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_subtask_policy = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        if not args.subtask_policy_use_hypernet:
            self.fc2_subtask_policy = nn.Linear(args.rnn_hidden_dim, args.max_subtasks * args.n_actions)
        else:
            self.fc2_w = nn.Linear(args.agent_subtask_embed_dim, args.rnn_hidden_dim * args.n_actions)
            self.fc2_b = nn.Linear(args.agent_subtask_embed_dim, args.n_actions)
        
        # 使用优化版本
        self.dynamic_subtask_generator = OptimizedDynamicSubtaskGenerator(args, args.agent_subtask_embed_dim)
        
        # 预计算掩码模板，避免重复创建
        self.register_buffer('mask_template', torch.zeros(1, 1, args.max_subtasks))
        
        # 预计算one-hot矩阵
        self.register_buffer('subtask_one_hot', 
                           torch.eye(args.max_subtasks).unsqueeze(0))

    def init_hidden_subtask_policy(self):
        return self.fc1_subtask_policy.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_hidden_agent_embed(self):
        return self.fc1_agent_embed.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state_subtask_policy, hidden_state_agent_embed, state, test_mode=False):
        # agent embedding
        x_agent_embed = F.relu(self.fc1_agent_embed(inputs))
        h_in_agent_embed = hidden_state_agent_embed.reshape(-1, self.args.rnn_hidden_dim)
        h_agent_embed = self.rnn_agent_embed(x_agent_embed, h_in_agent_embed)
        agent_embed = self.fc2_agent_embed(h_agent_embed).reshape(-1, self.args.n_agents, self.args.agent_subtask_embed_dim)

        # 优化：减少调用频率
        if self.training or (hasattr(self, '_step_count') and self._step_count % 5 == 0):
            n_subtasks = self.dynamic_subtask_generator(agent_embed, state, self.training)
            self._cached_n_subtasks = n_subtasks
        else:
            n_subtasks = getattr(self, '_cached_n_subtasks', self.args.max_subtasks)
        
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1

        # subtask representation - 使用预计算的one-hot
        bs = agent_embed.shape[0]
        subtask_one_hot = self.subtask_one_hot.expand(bs, -1, -1)
        subtask_embed = self.subtask_embed_net(subtask_one_hot)
        if self.args.use_tanh:
            subtask_embed = F.tanh(subtask_embed)

        # subtask policy
        x_subtask_policy = F.relu(self.fc1_subtask_policy(inputs))
        h_in_subtask_policy = hidden_state_subtask_policy.reshape(-1, self.args.rnn_hidden_dim)
        h_subtask_policy = self.rnn_subtask_policy(x_subtask_policy, h_in_subtask_policy)
        
        if not self.args.subtask_policy_use_hypernet:
            q = self.fc2_subtask_policy(h_subtask_policy).reshape(-1, self.args.max_subtasks, self.args.n_actions)
        else:
            subtask_embed_detach = subtask_embed.clone().detach()[0]
            w2 = self.fc2_w(subtask_embed_detach)
            b2 = self.fc2_b(subtask_embed_detach)
            w2 = w2.unsqueeze(0).expand(bs * self.args.n_agents, -1, -1).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
            b2 = b2.unsqueeze(0).expand(bs * self.args.n_agents, -1, -1).reshape(-1, 1, self.args.n_actions)
            h_subtask_policy_ = h_subtask_policy.unsqueeze(1).expand(-1, self.args.max_subtasks, -1).reshape(-1, 1, self.args.rnn_hidden_dim)
            q = th.bmm(h_subtask_policy_, w2) + b2
            q = q.reshape(-1, self.args.max_subtasks, self.args.n_actions)

        # subtask selection - 优化掩码操作
        subtask_prob_logit = th.bmm(agent_embed, subtask_embed.permute(0, 2, 1))
        
        # 优化的掩码操作
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