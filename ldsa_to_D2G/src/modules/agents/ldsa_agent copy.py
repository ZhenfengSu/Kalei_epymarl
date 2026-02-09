import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

class DynamicSubtaskGenerator(nn.Module):
    def __init__(self, args, embed_dim):
        super(DynamicSubtaskGenerator, self).__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.state_dims = args.state_dims
        
        # max_subtasks and min_subtasks
        self.max_subtasks = getattr(args, 'max_subtasks', 5)
        self.min_subtasks = getattr(args, 'min_subtasks', 2)
        
        # 可选的子任务数量范围
        self.num_subtask_options = self.max_subtasks - self.min_subtasks + 1
        
        # 上下文融合网络
        self.context_fusion = nn.Sequential(
            nn.Linear(embed_dim + args.state_dims, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        ) 
        
        # 子任务数量预测器 - 输出每个可能数量的logits
        self.subtask_count_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(getattr(args, 'dropout', 0.1)),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.num_subtask_options)  # 输出 [min_subtasks, max_subtasks] 范围内的logits
        )
        
        # Gumbel Softmax 温度参数
        self.temperature = getattr(args, 'gumbel_temperature', 1.0)
        
    def forward(self, agent_features, global_context=None, training=True):
        assert global_context is not None and self.context_fusion is not None, "Global context provided or context fusion network is not defined."
        batch_size = agent_features.shape[0]
        
        # 1. 聚合智能体特征
        aggregated_features = agent_features.mean(dim=1)  # [batch_size, embed_dim]
        
        # 2. 融合全局上下文
        if global_context is not None and self.context_fusion is not None:
            context_features = torch.cat([aggregated_features, global_context], dim=-1)
            fused_features = self.context_fusion(context_features)
        else:
            fused_features = aggregated_features
        
        # 3. 预测子任务数量的logits
        subtask_count_logits = self.subtask_count_predictor(fused_features)  # [batch_size, num_subtask_options]
        
        # 4. 使用Gumbel Softmax选择子任务数量
        if training:
            # 训练时使用Gumbel Softmax进行可微分采样
            subtask_count_probs = F.gumbel_softmax(
                subtask_count_logits, 
                tau=self.temperature, 
                hard=True,  # 使用hard模式获得one-hot向量
                dim=-1
            )  # [batch_size, num_subtask_options]
        else:
            # 测试时直接选择概率最大的
            subtask_count_probs = torch.zeros_like(subtask_count_logits)
            max_indices = torch.argmax(subtask_count_logits, dim=-1)
            subtask_count_probs.scatter_(1, max_indices.unsqueeze(1), 1.0)
        
        # 5. 计算实际的子任务数量
        # subtask_count_probs 是 one-hot 向量，我们需要将其转换为实际数量
        subtask_indices = torch.arange(
            self.min_subtasks, 
            self.max_subtasks + 1, 
            device=agent_features.device,
            dtype=torch.float32
        ).unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_subtask_options]
        
        # 计算期望的子任务数量（对于训练时的软分配）
        expected_n_subtasks = (subtask_count_probs * subtask_indices).sum(dim=-1)  # [batch_size]
        # 对于返回值，我们需要一个确定的整数
        if training:
            # 训练时返回期望值的整数部分（或者最可能的值）
            n_subtasks = torch.round(expected_n_subtasks).int().max().item()
        else:
            # 测试时返回选中的确切值
            n_subtasks = int(subtask_indices[subtask_count_probs.bool()].item())
        
        # 确保在有效范围内
        n_subtasks = max(self.min_subtasks, min(n_subtasks, self.max_subtasks))
        
        return n_subtasks

class LDSAAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LDSAAgent, self).__init__()
        self.args = args

        # agent embedding
        self.fc1_agent_embed = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_agent_embed = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_agent_embed = nn.Linear(args.rnn_hidden_dim, args.agent_subtask_embed_dim)

        # subtask representation
        if args.subtask_repr_layers == 2:
            self.subtask_embed_net = nn.Sequential(nn.Linear(args.max_subtasks, args.agent_subtask_embed_dim),
                                            nn.ReLU(),
                                            nn.Linear(args.agent_subtask_embed_dim, args.agent_subtask_embed_dim))
        elif args.subtask_repr_layers == 1:
            self.subtask_embed_net = nn.Linear(args.max_subtasks, args.agent_subtask_embed_dim, bias=False)
        
        # subtask policy
        self.fc1_subtask_policy = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_subtask_policy = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        if not args.subtask_policy_use_hypernet:
            self.fc2_subtask_policy = nn.Linear(args.rnn_hidden_dim, args.max_subtasks * args.n_actions)
        else:
            self.fc2_w = nn.Linear(args.agent_subtask_embed_dim, args.rnn_hidden_dim * args.n_actions)
            self.fc2_b = nn.Linear(args.agent_subtask_embed_dim, args.n_actions)
        
        self.dynamic_subtask_generator = DynamicSubtaskGenerator(args, args.agent_subtask_embed_dim)

    def init_hidden_subtask_policy(self):
        # make hidden states on same device as model
        return self.fc1_subtask_policy.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_hidden_agent_embed(self):
        # make hidden states on same device as model
        return self.fc1_agent_embed.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state_subtask_policy, hidden_state_agent_embed, state, test_mode=False):
        # inputs: [bs*n_agents, input_shape]
        # subtask_embed_input: [bs*n_subtasks, n_subtasks]
        
        # agent embedding
        x_agent_embed = F.relu(self.fc1_agent_embed(inputs))
        h_in_agent_embed = hidden_state_agent_embed.reshape(-1, self.args.rnn_hidden_dim)
        h_agent_embed = self.rnn_agent_embed(x_agent_embed, h_in_agent_embed)
        agent_embed = self.fc2_agent_embed(h_agent_embed).reshape(-1, self.args.n_agents, self.args.agent_subtask_embed_dim) # [bs, n_agents, embed_dim]

        # add
        n_subtasks = self.dynamic_subtask_generator(agent_embed, state)

        # subtask representation
        bs = agent_embed.shape[0]
        subtask_one_hot = th.eye(self.args.max_subtasks, device=inputs.device).unsqueeze(0).expand(bs, -1, -1) # [bs, n_subtasks, n_subtasks]
        subtask_embed = self.subtask_embed_net(subtask_one_hot) # [bs, n_subtasks, embed_dim]
        if self.args.use_tanh:
            subtask_embed = F.tanh(subtask_embed)

        # subtask policy
        x_subtask_policy = F.relu(self.fc1_subtask_policy(inputs))
        h_in_subtask_policy = hidden_state_subtask_policy.reshape(-1, self.args.rnn_hidden_dim)
        h_subtask_policy = self.rnn_subtask_policy(x_subtask_policy, h_in_subtask_policy)  # [bs*n_agents, rnn_hidden_dim]
        if not self.args.subtask_policy_use_hypernet:
            q = self.fc2_subtask_policy(h_subtask_policy).reshape(-1, self.args.max_subtasks, self.args.n_actions) # [bs*n_agents, n_subtasks, n_actions]
        else:
            subtask_embed_detach = subtask_embed.clone().detach()[0]  # [n_subtasks, embed_dim]
            w2 = self.fc2_w(subtask_embed_detach) # [n_subtasks, rnn_hidden_dim*n_actions]
            b2 = self.fc2_b(subtask_embed_detach) # [n_subtasks, n_actions]
            w2 = w2.unsqueeze(0).expand(bs * self.args.n_agents, -1, -1).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions) # [bs*n_agents*n_subtasks, rnn_hidden_dim, n_actions]
            b2 = b2.unsqueeze(0).expand(bs * self.args.n_agents, -1, -1).reshape(-1, 1, self.args.n_actions) # [bs*n_agents*n_subtasks, 1, n_actions]
            h_subtask_policy_ = h_subtask_policy.unsqueeze(1).expand(-1, self.args.max_subtasks, -1).reshape(-1, 1, self.args.rnn_hidden_dim) # [bs*n_agents*n_subtasks, 1, rnn_hidden_dim]
            q = th.bmm(h_subtask_policy_, w2) + b2
            q = q.reshape(-1, self.args.max_subtasks, self.args.n_actions) # [bs*n_agents, n_subtasks, n_actions]

        # subtask selection
        subtask_prob_logit = th.bmm(agent_embed, subtask_embed.permute(0, 2, 1)) # [bs, n_agents, n_subtasks]
        # add
        mask = th.zeros_like(subtask_prob_logit)
        mask[..., n_subtasks:] = -float('inf') 
        subtask_prob_logit = subtask_prob_logit + mask 
        if self.args.random_sele:
            subtask_prob_logit = th.rand_like(subtask_prob_logit)
        if test_mode and self.args.test_argmax:
            prob_max = th.max(subtask_prob_logit, dim=-1, keepdim=True)[1]
            subtask_prob = th.zeros_like(subtask_prob_logit).scatter_(-1, prob_max, 1)
        else:
            if self.args.sft_way == "softmax":
                subtask_prob = F.softmax(subtask_prob_logit, dim=-1) # [bs, n_agents, n_subtasks]
            elif self.args.sft_way == "gumbel_softmax":
                subtask_prob = F.gumbel_softmax(subtask_prob_logit, hard=True, dim=-1)
        subtask_prob = subtask_prob.reshape(-1, 1, self.args.max_subtasks) # [bs*n_agents, 1, n_subtasks]
        if self.args.evaluate:
            print('chosen_subtask_prob', subtask_prob.reshape(self.args.n_agents, self.args.max_subtasks))
        q = th.bmm(subtask_prob, q).squeeze(1) # [bs*n_agents, n_actions]

        return q, h_subtask_policy, h_agent_embed, subtask_prob_logit, subtask_embed 

