# Kaleidoscope 算法实现文档

本文档记录了在 `epymarl_kaleidoscope` 框架中实现 Kaleidoscope 算法的详细内容。

## 概述

Kaleidoscope 是一个通过非结构化剪枝（Unstructured Pruning）实现多智能体异质性的MARL算法。该算法通过为每个智能体学习不同的稀疏掩码（Sparse Masks），使智能体在共享参数的基础上实现功能专化。

## 算法核心思想

1. **非结构化剪枝**: 为每个神经网络的权重添加可学习的阈值参数，实现连接级别的剪枝
2. **智能体异质性**: 不同智能体拥有不同的稀疏掩码，从而实现不同的功能专化
3. **多样性损失**: 通过最小化不同智能体掩码的相似性，鼓励形成多样化的行为模式
4. **动态掩码重置**: 定期重置被剪枝的连接，促进探索新的网络结构

## 文件结构

### 新增文件

```
epymarl_kaleidoscope/
├── src/
│   ├── modules/
│   │   └── agents/
│   │       └── kalei_rnn_agent.py           # Kalei_RNNAgent_1R3 实现
│   ├── controllers/
│   │   └── kalei_controller.py              # Kalei_MAC 控制器
│   ├── learners/
│   │   └── kalei_q_learner.py               # Kalei_QLearner 学习器
│   └── config/
│       ├── algs/
│       │   └── Kalei_qmix_rnn_1R3.yaml      # 算法配置文件
│       └── envs/
│           └── mpe.yaml                      # MPE环境配置
├── run_kalei_mpe.py                          # Python运行脚本
└── run_kalei_mpe.sh                          # Shell运行脚本
```

### 修改文件

- `src/modules/agents/__init__.py`: 注册 `Kalei_RNNAgent_1R3`
- `src/controllers/__init__.py`: 注册 `Kalei_MAC`
- `src/learners/__init__.py`: 注册 `Kalei_QLearner`

## 详细代码说明

### 1. kalei_rnn_agent.py

核心组件包括：

#### compare_STE 类
```python
class compare_STE(th.autograd.Function):
```
- Straight-Through Estimator (直通估计器)
- 使二值化操作可微分
- 前向传播：符号函数 + ReLU
- 反向传播：tanh 梯度

#### KaleiLinear 类
```python
class KaleiLinear(nn.Linear):
```
- 继承 `nn.Linear`，添加稀疏掩码功能
- **核心参数**:
  - `sparse_thresholds`: 可学习的阈值参数 (shape: `[n_masks, out_features, in_features]`)
  - `n_masks`: 智能体数量（每个智能体一个掩码）
  - `threshold_init_scale/bias`: 阈值初始化参数（负值鼓励初始稀疏性）
  - `threshold_reset_scale/bias`: 阈值重置参数

- **核心方法**:
  - `_sparse_function()`: 应用阈值进行剪枝，输出 `sign(w) * relu(|w| - sigmoid(threshold))`
  - `_get_masks()`: 获取二值化掩码（用于多样性损失计算）
  - `get_sparsities()`: 计算稀疏度统计
  - `_reset()`: 重置已剪枝连接的权重和阈值

#### Kalei_RNNAgent_1R3 类
```python
class Kalei_RNNAgent_1R3(nn.Module):
```
- 1R3 架构：1个RNN层 + 3个隐藏层
- **网络结构**: `fc1 -> rnn -> fc2 -> fc3 -> fc4`
- 所有线性层使用 `KaleiLinear`（RNN除外）
- **关键方法**:
  - `forward(inputs, hidden_state, agent_ids)`: 接收智能体ID，应用对应的稀疏掩码
  - `mask_diversity_loss()`: 计算多样性损失，鼓励不同智能体的掩码不同
  - `_reset_all_masks_weights(reset_ratio)`: 重置指定比例的掩码

### 2. kalei_controller.py

```python
class Kalei_MAC(BasicMAC):
```
- 继承 `BasicMAC`
- **新增属性**:
  - `sparsities`: 获取稀疏度统计
  - `mask_parameters`: 获取掩码阈值参数
- **关键修改**:
  - `_build_inputs()`: 返回 `inputs` 和 `agent_ids`
  - `forward()`: 传递 `agent_ids` 给 agent，用于选择对应的稀疏掩码

### 3. kalei_q_learner.py

```python
class Kalei_QLearner(QLearner):
```
- 继承 `QLearner`
- **核心功能**:
  - **多样性损失**: 调用 `mask_diversity_loss()` 计算智能体间的掩码差异
  - **自适应多样性系数**: 根据历史 TD loss 和多样性损失的比例动态调整
  - **掩码重置**: 每 `reset_interval` 步重置一次掩码（但不在训练结束时）
  - **梯度控制**: 训练时启用掩码参数梯度

- **损失函数**:
  ```python
  loss = td_loss + div_coef * div_loss
  ```
  - `td_loss`: 标准的时序差分损失
  - `div_loss`: 多样性损失（负值，最大化掩码差异）
  - `div_coef`: 自适应系数，平衡两种损失

### 4. 配置文件: Kalei_qmix_rnn_1R3.yaml

```yaml
# QMIX 基础配置
agent: "kalei_rnn_1R3"
mac: "kalei_mac"
learner: "kalei_q_learner"
mixer: "qmix"

# Kaleidoscope 特定参数
Kalei_args:
  deque_len: 100                    # 损失历史队列长度
  div_coef: 0.5                     # 多样性损失基础系数
  threshold_init_scale: 5.0         # 阈值初始化缩放
  threshold_init_bias: 5.0          # 阈值初始化偏置
  threshold_reset_scale: 5.0        # 阈值重置缩放
  threshold_reset_bias: 5.0         # 阈值重置偏置
  reset_interval: 200_000           # 掩码重置间隔（步数）
  reset_ratio: 0.1                  # 重置比例
  sparsity_layer_weights:           # 各层稀疏度权重
    - 1.0   # fc1
    - 2.0   # fc2
    - 4.0   # fc3
    - 8.0   # fc4
```

**参数说明**:
- `deque_len`: 用于计算自适应多样性系数的历史损失数量
- `div_coef`: 基础多样性系数，实际使用时会根据损失历史自适应调整
- `threshold_init_*`: 初始阈值设置为负值（-5到-10），鼓励初始时更多连接被剪枝
- `threshold_reset_*`: 重置时同样使用负阈值
- `reset_interval`: 每20万步重置一次掩码，促进探索
- `reset_ratio`: 只重置10%的被剪枝连接
- `sparsity_layer_weights`: 越深的层权重越大，鼓励深层网络的稀疏性差异

## MPE 环境支持

### MPE 环境列表

epymarl 框架通过 `gymma` 接口支持 MPE 环境。常用 MPE 环境：

1. **simple_spread_v3**: 合作导航任务
   - N个智能体需要覆盖N个地标
   - 智能体需要分散开来，避免碰撞

2. **simple_tag_v3**: 捕食者-猎物任务
   - 多个捕食者智能体追逐猎物
   - 猎物移动速度更快但数量少

3. **simple_adversary_v3**: 对抗任务
   - 1个对抗者智能体 vs N个正常智能体
   - 目标相反

4. **simple_crypto_v3**: 通信任务
   - 智能体需要通过通信传递信息

5. **simple_push_v3**: 合作推箱子
   - 智能体合作推开障碍物

6. **simple_reference_v3**: 参考任务
   - 智能体需要参考同伴的观测

### 配置文件: mpe.yaml

```yaml
env: "gymma"
env_args:
  key: null                    # 环境名称（运行时指定）
  time_limit: 25              # 每回合最大步数
  common_reward: True         # 共享奖励
  reward_scalarisation: "sum" # 奖励聚合方式
```

## 使用方法

### 方法1: 使用 Python 脚本

```bash
# 基本用法
python run_kalei_mpe.py --env-name simple_spread_v3

# 自定义参数
python run_kalei_mpe.py \
    --env-name simple_tag_v3 \
    --t-max 1000000 \
    --batch-size-run 32 \
    --log-dir results/my_experiment

# 评估模式
python run_kalei_mpe.py \
    --env-name simple_spread_v3 \
    --evaluate \
    --load-ckpt results/mpe/Kalei_qmix_rnn_1R3/simple_spread_v3/models/
```

### 方法2: 使用 Shell 脚本

```bash
# 运行默认环境 (simple_spread_v3)
./run_kalei_mpe.sh

# 指定环境
./run_kalei_mpe.sh simple_tag_v3

# 其他环境
./run_kalei_mpe.sh simple_adversary_v3
./run_kalei_mpe.sh simple_crypto_v3
```

### 方法3: 直接使用 main.py

```bash
python src/main.py \
    --env-config gymma \
    --env-args key=simple_spread_v3 time_limit=25 common_reward=True \
    --config Kalei_qmix_rnn_1R3 \
    --experiment-name my_experiment
```

## 训练输出

### 日志记录

训练过程会记录以下指标：

- **损失指标**:
  - `loss_td`: 时序差分损失
  - `div_loss`: 多样性损失
  - `loss`: 总损失
  - `div_coef`: 自适应多样性系数

- **性能指标**:
  - `td_error_abs`: 平均TD误差
  - `q_taken_mean`: 平均Q值
  - `target_mean`: 平均目标Q值

- **稀疏度指标**:
  - `sparsity_layer_0` to `sparsity_layer_3`: 各层稀疏度
  - `overall_sparsity`: 整体稀疏度

### 结果保存

```
results/mpe/
└── Kalei_qmix_rnn_1R3/
    └── simple_spread_v3/
        ├── models/           # 模型检查点
        ├── logs/             # TensorBoard日志
        └── recap/            # 训练总结
```

## 算法流程

### 训练循环

```python
for episode in training:
    # 1. 收集经验
    for t in episode:
        agent_ids = [0, 1, 2, ...]  # 智能体ID
        actions = mac.forward(obs, agent_ids)
        next_obs, rewards, dones = env.step(actions)

    # 2. 训练（每batch_size_run个episode）
    if should_train():
        # 2.1 启用掩码梯度
        mac.agent.set_require_grads(mode=True)

        # 2.2 检查是否需要重置掩码
        if t_env - last_reset_t > reset_interval:
            mac.agent._reset_all_masks_weights(reset_ratio)

        # 2.3 计算损失
        td_loss = compute_td_loss()  # 标准Q学习损失
        div_loss = mac.agent.mask_diversity_loss()  # 多样性损失

        # 2.4 自适应系数
        div_coef = abs(div_coef * mean(td_loss_history) / mean(div_loss_history))

        # 2.5 反向传播
        loss = td_loss + div_coef * div_loss
        loss.backward()
        optimizer.step()
```

### 推理循环

```python
# 推理时禁用掩码梯度
mac.agent.set_require_grads(mode=False)

agent_outs = mac.forward(obs, agent_ids)  # 使用学习到的稀疏掩码
actions = select_actions(agent_outs)
```

## 关键技术细节

### 1. 阈值机制

每个智能体在每个连接上有一个独立的阈值：
```python
# 初始化：负阈值鼓励剪枝
threshold = -(rand(0,1) * scale + bias)  # 例如: -5到-10

# 前向传播：剪枝
mask = sigmoid(threshold)
w_pruned = sign(w) * relu(|w| - mask)

# 如果 |w| < sigmoid(threshold)，则 w_pruned = 0
```

### 2. 直通估计器 (STE)

```python
# 前向：离散化
def forward(ctx, input):
    return relu(sign(input))

# 反向：连续梯度
def backward(ctx, grad_output):
    return tanh(grad_output)
```

这使得二值化操作可以反向传播梯度。

### 3. 多样性损失计算

```python
# 对于每一层：
m = get_weighted_masks(weights, agent_ids)  # shape: [n_agents, out_dim, in_dim]

# 计算所有智能体对的差异
diversity = sum(|m[i] - m[j]|) for all i, j

# 负号：最大化多样性
loss = -diversity / (n_agents * (n_agents - 1)) / n_params
```

### 4. 掩码重置机制

```python
# 只重置完全被剪枝的连接
fully_pruned = (sum(masks) == 0)  # 所有智能体的掩码都为0

# 随机选择其中10%重置
reset_flag = (rand() < reset_ratio) AND fully_pruned

# 重置阈值和权重
new_thresholds = reset_flag * random_init() + (1-reset_flag) * old_thresholds
new_weights = reset_flag * kaiming_init() + (1-reset_flag) * old_weights
```

## 调试和优化建议

### 调试技巧

1. **监控稀疏度**: 观察各层稀疏度是否在合理范围（30%-80%）
2. **检查多样性损失**: 应该是负值，且绝对值不应过大
3. **验证掩码差异**: 不同智能体的稀疏模式应该有明显差异

### 超参数调优

1. **div_coef**: 控制异质性强度
   - 太小：智能体行为趋同
   - 太大：可能损害性能

2. **reset_interval**: 控制探索频率
   - 太短：难以收敛
   - 太长：探索不足

3. **threshold_init_bias**: 控制初始稀疏度
   - 负值越大：初始越稀疏
   - 建议范围：3.0 - 10.0

### 常见问题

**Q1: 训练不收敛怎么办？**
- 降低 `div_coef`
- 增加 `threshold_init_bias`（降低初始稀疏度）
- 检查学习率

**Q2: 智能体行为过于相似？**
- 提高 `div_coef`
- 增加 `sparsity_layer_weights`
- 检查多样性损失是否正确计算

**Q3: 某些智能体完全不作为？**
- 稀疏度过高，降低 `threshold_init_bias`
- 增加 `reset_interval`，给更多时间学习
- 检查奖励函数是否合理

## 参考文献

Kaleidoscope 算法基于以下研究：
- 通过非结构化剪枝实现多智能体异质性
- 动态网络结构探索
- 稀疏表示在MARL中的应用

## 总结

本实现将 Kaleidoscope 算法完整移植到 epymarl 框架，支持在 MPE 等 Gym-Multi-Agent 环境上运行。核心创新在于：

1. **KaleiLinear**: 可学习的稀疏线性层
2. **多样性损失**: 鼓励智能体异质性
3. **自适应系数**: 平衡性能和多样性
4. **动态重置**: 持续探索新结构

该实现保持了与原始算法的一致性，同时充分利用了 epymarl 框架的模块化设计。
