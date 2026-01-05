# K-2:4 算法使用指南

## 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12 (支持 NVIDIA Ampere 架构以获得最佳性能)
- CUDA >= 11.0 (用于半结构化稀疏硬件加速)

### 安装依赖

```bash
cd epymarl_kaleidoscope
pip install -r requirements.txt

# 或对于 SMACv2
cd Kaleidoscope/Kalei_SMACv2/src
pip install -r requirements.txt
```

---

## 配置说明

### 基本配置

在您的配置文件中添加 K-2:4 特定参数:

```python
# config.py

# 基本参数
args.n_agents = 3  # 智能体数量
args.hidden_dim = 64  # 隐藏层维度
args.n_actions = 6  # 动作空间大小
args.use_rnn = True  # 是否使用 RNN

# K-2:4 算法参数
args.K24_args = {
    # === 温度退火参数 ===
    "temperature_init": 5.0,      # 初始温度(探索阶段)
    "temperature_min": 0.1,       # 最小温度(收敛阶段)
    "anneal_start": 0.0,          # 开始退火的训练进度(0.0-1.0)
    "anneal_end": 0.8,            # 完成退火的训练进度(0.0-1.0)
    "anneal_end_step": 800000,    # 退火结束步数(通常为 t_max * 0.8)

    # === EMA 激活追踪 ===
    "ema_momentum": 0.99,         # EMA 动量(0.9-0.999,越高越平滑)

    # === 异构系数初始化 ===
    "hetero_init_scale": 0.01,    # α 系数初始化标准差

    # === 多样性损失 ===
    "div_coef": 0.1,              # 基础多样性系数(0.01-1.0)
    "deque_len": 100,             # 损失历史队列长度(用于自适应系数)

    # === 自适应重置 ===
    "reset_interval": 10000,      # 定期重置间隔(环境步数)
    "reset_ratio": 0.1,           # 重置比例(0.05-0.2,即 5%-20%)
    "use_adaptive_reset": False,  # 是否启用基于 KL 散度的自适应重置
    "kl_threshold": 0.1,          # KL 散度阈值(触发重置的条件)
}
```

### 参数调优建议

| 参数 | 默认值 | 调优范围 | 效果 |
|------|--------|---------|------|
| `temperature_init` | 5.0 | 3.0-10.0 | 越高探索越强,收敛越慢 |
| `temperature_min` | 0.1 | 0.01-0.5 | 越低最终策略越确定 |
| `div_coef` | 0.1 | 0.01-1.0 | 越高智能体差异越大,但可能降低性能 |
| `reset_interval` | 10000 | 5000-50000 | 越小重置越频繁,探索越多 |
| `reset_ratio` | 0.1 | 0.05-0.2 | 越大每次重置越多,可能不稳定 |

---

## 训练命令

### epymarl_kaleidoscope

```bash
# 使用 K-2:4 算法训练
python src/main.py --config=corruption --env-config=mpe_simple_spread \
    --alg-name=K24QMix \
    --n-agents=3 \
    --hidden-dim=64

# SMAC 环境
python src/main.py --config=corruption --env-config=smac \
    --alg-name=K24QMix \
    --env-args.map_name="3m" \
    --K24_args.temperature_init=5.0 \
    --K24_args.div_coef=0.1
```

### Kalei_SMACv2

```bash
# 训练 SMACv2 场景
python src/main.py --env=smac_v2 --map_name="3m" \
    --alg_name=k24_nq \
    --n_unit_types=3 \
    --K24_args.temperature_init=5.0 \
    --K24_args.div_coef=0.1
```

---

## 代码集成示例

### 创建自定义 Agent

```python
from modules.agents.k24_rnn_agent import K24_RNNAgent
from functools import partial

# 自定义配置
class CustomK24Agent(K24_RNNAgent):
    def __init__(self, input_shape, args):
        # 自定义 K-2:4 参数
        custom_args = types.SimpleNamespace(
            n_agents=args.n_agents,
            hidden_dim=args.hidden_dim,
            n_actions=args.n_actions,
            use_rnn=True,
            K24_args={
                "temperature_init": 3.0,  # 更低的初始温度
                "div_coef": 0.2,          # 更强的多样性
                "reset_interval": 5000,   # 更频繁的重置
            }
        )
        super().__init__(input_shape, custom_args)

        # 可以添加自定义层
        self.custom_layer = nn.Linear(args.hidden_dim, args.hidden_dim)
```

### 自定义 Learner

```python
from learners.k24_q_learner import K24_QLearner

class CustomK24Learner(K24_QLearner):
    def train(self, batch, t_env, episode_num):
        # 调用原始训练方法
        super().train(batch, t_env, episode_num)

        # 添加自定义逻辑
        if t_env % 1000 == 0:
            # 例如: 记录额外的统计信息
            custom_stats = self.compute_custom_stats()
            self.logger.log_stat("custom_metric", custom_stats, t_env)

    def compute_custom_stats(self):
        # 计算自定义统计量
        pattern_stats = self.mac.agent.get_pattern_stats()
        return pattern_stats.get('pattern_entropy', 0.0)
```

---

## 监控和调试

### 关键指标

训练过程中会记录以下关键指标:

#### 1. 损失指标
- `loss_td`: TD 损失(Q-learning 误差)
- `div_loss`: 多样性损失(越低越好,表示智能体差异越大)
- `div_coef`: 自适应多样性系数(自动调整)
- `loss`: 总损失 = td_loss + div_coef * div_loss

#### 2. 温度和进度
- `temperature`: 当前温度(从 5.0 退火到 0.1)
- `progress`: 训练进度(0.0 到 1.0)

#### 3. 模式统计
- `pattern_mean_similarity`: 智能体间平均模式相似度(越低越好)
- `pattern_max_similarity`: 最大相似度(应 < 0.5)
- `pattern_min_similarity`: 最小相似度
- `pattern_0_prob` 到 `pattern_5_prob`: 各模式使用概率
- `pattern_entropy`: 模式分布熵(越高越均匀)

#### 4. 稀疏度统计
- `sparsity_layer_0` 到 `sparsity_layer_3`: 各层稀疏度(应约为 0.5)
- `overall_sparsity`: 整体网络稀疏度(应约为 0.5)

### TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir=results/

# 在浏览器中访问 http://localhost:6006
```

### 检查点和分析

```python
# 加载检查点并分析
import torch
from modules.agents.k24_rnn_agent import K24_RNNAgent

# 加载模型
checkpoint = torch.load("results/model.pth")
agent = checkpoint["agent"]

# 分析模式分布
pattern_stats = agent.get_pattern_stats()
print("模式使用概率:", pattern_stats["pattern_mean"])
print("模式熵:", pattern_stats["pattern_entropy"])

# 分析稀疏度
sparsities, _, overall_sparsity = agent.get_sparsities()
print("各层稀疏度:", sparsities)
print("整体稀疏度:", overall_sparsity)
```

---

## 常见问题

### Q1: 训练损失不下降

**可能原因**:
- `div_coef` 过高,多样性损失主导了训练
- 温度过高,探索过度

**解决方案**:
```python
# 降低多样性系数
args.K24_args["div_coef"] = 0.05

# 或降低初始温度
args.K24_args["temperature_init"] = 3.0
```

### Q2: 智能体策略趋于相同

**可能原因**:
- `div_coef` 过低,多样性约束不足

**解决方案**:
```python
# 提高多样性系数
args.K24_args["div_coef"] = 0.2

# 或启用更频繁的重置
args.K24_args["reset_interval"] = 5000
```

### Q3: 训练不稳定

**可能原因**:
- `reset_ratio` 过高,每次重置太多参数

**解决方案**:
```python
# 降低重置比例
args.K24_args["reset_ratio"] = 0.05

# 或启用自适应重置
args.K24_args["use_adaptive_reset"] = True
```

### Q4: 稀疏度不是 50%

**可能原因**:
- 权重维度不是 4 的倍数
- 实现中的 padding 逻辑问题

**解决方案**:
```python
# 确保 hidden_dim 是 4 的倍数
args.hidden_dim = 64  # 正确
args.hidden_dim = 65  # 可能导致稀疏度不是 50%
```

---

## 性能优化建议

### 1. GPU 加速

```python
# 确保 model 在正确的设备上
model = model.to(device)

# 使用混合精度训练(需要 GPU 支持 Tensor Core)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = learner.train(batch, t_env, episode_num)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 批处理优化

```python
# 增加批大小以充分利用 GPU
args.batch_size = 32  # 根据显存调整
args.buffer_size = 5000
```

### 3. EMA 追踪优化

```python
# 减少 EMA 更新频率(每 k 步更新一次)
if step % k == 0:
    layer.ema_tracker.update(activation)
```

---

## 与原版 Kaleidoscope 对比实验

### 实验设置

```python
# 原版 Kaleidoscope
kalei_args = {
    "threshold_init_scale": 0.5,
    "threshold_init_bias": 0.0,
    "div_coef": 0.1,
    "reset_interval": 10000,
}

# K-2:4 版本
K24_args = {
    "temperature_init": 5.0,
    "div_coef": 0.1,
    "reset_interval": 10000,
}
```

### 预期结果

| 指标 | Kaleidoscope | K-2:4 |
|------|-------------|-------|
| **训练时间** | 1x | 0.9-1.0x |
| **推理速度** | 1x | 1.5-2.0x |
| **最终性能** | 基准 | 相当或略优 |
| **策略多样性** | 基准 | 相当 |
| **内存占用** | 1x | 1.0-1.1x |

---

## 扩展和定制

### 添加新的多样性损失

```python
# 在 k24_diversity.py 中添加
class CustomDiversityLoss(nn.Module):
    def forward(self, pattern_probs_list):
        # 自定义损失计算
        # 例如: 基于模式的 KL 散度
        loss = 0
        for probs in pattern_probs_list:
            # 计算均匀分布
            uniform = th.ones_like(probs) / 6
            # KL 散度
            kl = (probs * th.log(probs / uniform)).sum()
            loss += kl
        return loss

# 在 learner 中使用
self.custom_diversity = CustomDiversityLoss()
```

### 自定义温度调度

```python
# 在 k24_rnn_agent.py 中修改
def custom_anneal_temperature(self, progress):
    # 余弦退火
    import math
    temp = self.temperature_min + 0.5 * (self.temperature_init - self.temperature_min) * \
           (1 + math.cos(math.pi * progress))

    for layer in self.mask_layers:
        layer.set_temperature(temp)
```

### 支持不同的稀疏模式

```python
# 在 k24_pattern_gumbel_layer.py 中扩展
class Pattern4x8Matrix:
    """支持 4:8 稀疏模式"""
    def __init__(self, device):
        # 定义 4:8 的合法模式
        # ...
        pass
```

---

## 引用

如果您在研究中使用了 K-2:4 算法,请引用:

```bibtex
@article{k24_2024,
  title={K-2:4: Hardware-Friendly Semi-Structured Sparsity for
         Heterogeneous Multi-Agent Reinforcement Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 联系和支持

- GitHub Issues: [项目地址]
- 邮件: [您的邮箱]
- 文档: [完整文档链接]

---

## 许可证

本项目遵循原 Kaleidoscope 项目的许可证(MIT 或 Apache 2.0)。
