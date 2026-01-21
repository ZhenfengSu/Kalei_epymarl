基于提供的文章内容，这篇文章提出的核心方法是**基于结构化网络剪枝的参数共享 (Structured Network Pruning for Parameter Sharing, 简称 SNP-PS)**。

以下是该方法的详细逻辑和步骤总结：

**1. 理论基础：彩票组假设 (Lottery Group Ticket Hypothesis, LGTH)**
 作者受“彩票假设”（Lottery Ticket Hypothesis）启发，提出了一个新的猜想：在一个随机初始化且足够大的稠密网络中，存在一组子网络（Subnetworks），这些子网络能够让多个智能体具备可识别性（Identifiability），并且在性能上可以媲美甚至超越简单的全参数共享网络。

**2. 核心机制：结构化剪枝 (Structured Network Pruning)**
 为了找到上述的“中奖组彩票”（Winning Group Tickets），作者提出了 SNP-PS 方法：

- **初始化：** 首先构建一个随机初始化的稠密深度神经网络（作为根网络）。
- **生成掩码 (Mask Generation)：** 使用**结构化剪枝**（即整组移除与某个神经元相连的权重，而不是移除单个权重）。
- **独立剪枝：** 对同一个根网络进行 $N$ 次独立的随机剪枝，为 $N$ 个智能体生成 $N$ 个不同的二进制掩码（Binary Masks）。
- **子网络构建：** 每个智能体拥有自己的掩码 $M_i$，其策略或价值函数由 $f(x; \theta \odot M_i)$ 表示。这意味着每个智能体只使用根网络参数的一个子集。

**3. 运作方式：部分参数共享**

- 共享与独立并存：

   由于掩码是随机生成的，不同智能体的子网络在某些神经元上会重叠，在某些上则不重叠。

  - **重叠部分：** 实现了参数共享，利用所有共享该神经元的智能体的梯度进行训练，保持了高样本效率（Sample Efficiency）。
  - **不重叠部分：** 充当了智能体的独有参数，增加了联合策略的表达能力（Representational Capacity），使得智能体即使面对相同的观测也能做出不同的动作。

- **训练：** 在训练过程中，只需要维护一个稠密网络（根网络），通过中心化训练（CTDE）更新参数。

**4. 方法优势**
 与传统的“参数共享+One-Hot编码”或“选择性参数共享（SePS）”相比，SNP-PS 的核心优势在于：

- **无需额外参数：** 不需要引入 One-Hot 向量或额外的聚类网络，参数量与简单的全参数共享相同。
- **可控性：** 可以通过调整剪枝率（Pruning Ratio）来控制智能体之间参数共享的程度。
- **增强表现力：** 能够有效处理异构多智能体环境，允许智能体学习多样化的行为。







Github issue

# [Question] Does this include code for the baselines like SNP? #1

Open

[![@KaleabTessera](https://avatars.githubusercontent.com/u/10942061?u=0ab394ad1daedd0a96c07d92b7d5a77c7b115b27&v=4&size=80)](https://github.com/KaleabTessera)

## Description

[KaleabTessera](https://github.com/KaleabTessera)

opened [on Jan 4, 2025](https://github.com/LXXXXR/Kaleidoscope/issues/1#issue-2767973886)

Hi, thanks for open-sourcing your code!

I can't seem to find the code for SNP? Not sure how you ran experiments to compare with that?

## Activity

[![LXXXXR](https://avatars.githubusercontent.com/u/73265258?u=07843ebeb5cbf57685aa15005a8225dc408ba68c&v=4&size=80)](https://github.com/LXXXXR)

### LXXXXR commented on Jan 8, 2025

[LXXXXR](https://github.com/LXXXXR)

[on Jan 8, 2025](https://github.com/LXXXXR/Kaleidoscope/issues/1#issuecomment-2576722256)

Owner

Thank you for your interest in our work.

For baselines, we used official implementations whenever possible to ensure faithful reproduction. In the case of SNP, since we couldn't locate the official codebase, we implemented the method following the specifications in their paper. Here's the key part of our implementation of the SNP method:

```
class SNP_RNNAgent_1R3(RNNAgent_1R3):
    def __init__(self, *args, **kwargs):
        super(SNP_RNNAgent_1R3, self).__init__(*args, **kwargs)

        self.sparsity_ratios = self.args.SNP_args["layers_sparsities"]
        self.n_agents = self.args.n_agents
        assert len(self.sparsity_ratios) == 3
        for i, layer_sparsity in enumerate(self.sparsity_ratios):
            self.register_buffer(
                f"mask_{i}",
                th.rand(self.n_agents, self.args.hidden_dim) > layer_sparsity,
            )

    def forward(self, inputs, hidden_state, agent_ids):
        # agent dimention is indexed
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        agent_ids = agent_ids.reshape(-1)

        x = F.relu(self.fc1(inputs))

        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

        # alghough not specified in the paper,
        # structured pruning in general happens before the activation layer
        h = h * self.mask_0[agent_ids]
        q = self.fc2(h) * self.mask_1[agent_ids]
        q = F.relu(q)
        q = self.fc3(q) * self.mask_2[agent_ids]
        q = F.relu(q)
        q = self.fc4(q)

        return q.view(b, a, -1), h.view(b, a, -1)
```



Feel free to let me know if you have any other questions!

[![KaleabTessera](https://avatars.githubusercontent.com/u/10942061?u=0ab394ad1daedd0a96c07d92b7d5a77c7b115b27&v=4&size=80)](https://github.com/KaleabTessera)

### KaleabTessera commented on Jan 9, 2025

[KaleabTessera](https://github.com/KaleabTessera)

[on Jan 9, 2025](https://github.com/LXXXXR/Kaleidoscope/issues/1#issuecomment-2578285684)

Author

Thanks for the reply [@LXXXXR](https://github.com/LXXXXR) !!

How would I go about running this on an environment like MaMujoco?

[![LXXXXR](https://avatars.githubusercontent.com/u/73265258?u=07843ebeb5cbf57685aa15005a8225dc408ba68c&v=4&size=80)](https://github.com/LXXXXR)

### LXXXXR commented on Feb 17, 2025

[LXXXXR](https://github.com/LXXXXR)

[on Feb 17, 2025](https://github.com/LXXXXR/Kaleidoscope/issues/1#issuecomment-2663373066)

Owner

Happy to share the key part of our implementation of the SNP method with HARL repo:

```
class SNP_MLP(nn.Module):
    def __init__(self, sizes, activation_func, args, final_activation_func="identity"):
        super().__init__()

        self.sparsity_ratios = args["SNP_args"]["layers_sparsities"]
        self.n_masks = args["SNP_args"]["n_masks"]

        # need to assert, not general implementation
        assert len(sizes) == 4
        self.activation_func = get_active_func(activation_func)
        self.final_activation_func = get_active_func(final_activation_func)
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        self.fc3 = nn.Linear(sizes[2], sizes[3])

        for i, layer_sparsity in enumerate(self.sparsity_ratios):
            self.register_buffer(
                f"mask_{i}",
                th.rand(self.n_masks, sizes[i + 1]) > layer_sparsity,
            )

    def forward(self, x, mask_id):
        x = self.activation_func(self.fc1(x) * self.mask_0[mask_id])
        x = self.activation_func(self.fc2(x) * self.mask_1[mask_id])
        x = self.final_activation_func(self.fc3(x) * self.mask_2[mask_id])

        return x
```



While this implementation allowed us to reproduce the key mechanisms described in the SNP paper, please note that it may differ from their original implementation.