# K-2:4 半结构化稀疏异构多智能体强化学习

## 项目简介

本项目实现了基于 **K-2:4 (Kaleidoscope-2:4 via Pattern Gumbel)** 算法的异构多智能体强化学习框架。该算法是原始 Kaleidoscope 算法的改进版本,使用 **2:4 半结构化稀疏剪枝** 替代非结构化剪枝,在保持智能体异构性的同时,实现约 **2x 的推理加速**。

### 核心创新

1. **硬件友好**: 使用 2:4 半结构化稀疏,完美适配 NVIDIA Ampere+ Tensor Cores
2. **完全可微**: 通过 Gumbel-Softmax 实现端到端可微训练
3. **动态适应**: EMA 激活追踪解决 RL 无校准集问题
4. **模式正交**: 概率分布层面的异构性约束

---

## 文件结构

```
Semi_Kalei/
├── 最终方案设计_K_24.md                    # 算法设计方案文档
├── K-2_4_Implementation_Documentation.md   # 详细实现文档
├── K-2_4_User_Guide.md                     # 用户使用指南
├── README_K24.md                           # 本文件
│
├── epymarl_kaleidoscope/                   # 通用 MARL 框架实现
│   └── src/
│       ├── modules/
│       │   ├── layer/
│       │   │   ├── k24_pattern_gumbel_layer.py  # 核心: 模块 A + B
│       │   │   └── k24_diversity.py             # 核心: 模块 C
│       │   └── agents/
│       │       └── k24_rnn_agent.py            # K-2:4 Agent
│       └── learners/
│           └── k24_q_learner.py                # K-2:4 Learner
│
└── Kaleidoscope/Kalei_SMACv2/             # SMACv2 专门实现
    └── src/src/
        ├── modules/
        │   ├── layer/
        │   │   ├── k24_pattern_gumbel_layer.py  # 从 epymarl 复制
        │   │   └── k24_diversity.py
        │   └── agents/
        │       └── k24_rnn_agent.py            # SMACv2 版本
        └── learners/
            └── k24_nq_learner.py                # SMACv2 版本
```

---

## 快速开始

### 安装

```bash
# 克隆项目
cd /path/to/Semi_Kalei

# 安装 epymarl 版本
cd epymarl_kaleidoscope
pip install -r requirements.txt

# 或安装 SMACv2 版本
cd Kaleidoscope/Kalei_SMACv2/src
pip install -r requirements.txt
```

### 训练示例

```bash
# epymarl_kaleidoscope - MPE 环境
python src/main.py --config=corruption --env-config=mpe_simple_spread \
    --alg-name=K24QMix --n-agents=3

# Kalei_SMACv2 - StarCraft II
python src/main.py --env=smac_v2 --map_name="3m" \
    --alg_name=k24_nq --n_unit_types=3
```

---

## 核心模块说明

### 模块 A: 动态异构评分

**位置**: `src/modules/layer/k24_pattern_gumbel_layer.py`

**核心公式**:
```
S_i = |W_shared| · EMA(|A_i|) · σ(α_i)
```

**关键类**:
- `EMAActivationTracker`: 激活值指数移动平均
- `SemiStructuredLinear24._compute_heterogeneous_scores()`: 异构评分计算

**实现要点**:
- 使用 EMA 在线统计,无需离线校准集
- 每个智能体有独立的异构系数 `hetero_alpha`
- Sigmoid 调制确保系数在 (0, 1) 范围

---

### 模块 B: 模式投影

**位置**: `src/modules/layer/k24_pattern_gumbel_layer.py`

**核心机制**:
```
1. 模式打分: Logits = S_i × M^T
2. Gumbel-Softmax: π = GumbelSoftmax(Logits, τ)
3. 掩码重建: Mask = π × M
```

**关键类**:
- `Pattern24Matrix`: 定义 6 种合法 2:4 稀疏模式
- `SemiStructuredLinear24._pattern_gumbel_softmax()`: 模式采样

**6 种模式**:
```
[1,1,0,0] [1,0,1,0] [1,0,0,1] [0,1,1,0] [0,1,0,1] [0,0,1,1]
   0        1        2        3        4        5
```

---

### 模块 C: 模式正交

**位置**: `src/modules/layer/k24_diversity.py`

**核心公式**:
```
L_div = (1/N_groups) · Σ_g (π_A · π_B)
```

**关键类**:
- `PatternOrthogonalityLoss`: 模式正交损失计算
- `K24DiversityManager`: 多样性损失管理和自适应系数

**优势**:
- 比权重 L1 距离更直接
- 概率分布层面的约束
- 易于优化和解释

---

## 训练动态

### 温度退火

```
阶段 I (探索): τ = 5.0 → 1.0
  - 高噪声,掩码随机跳变
  - 帮助异构系数探索不同结构

阶段 II (收敛): τ = 1.0 → 0.1
  - 低噪声,分布趋向 one-hot
  - 消除训练-推理 gap
```

**代码位置**: `src/modules/agents/k24_rnn_agent.py:anneal_temperature()`

---

### 自适应重置

**触发机制**:
1. **定期重置**: 每 N 步(默认 10000)
2. **自适应重置**: KL 散度超过阈值(可选)

**重置操作**:
- 软重置 EMA 统计量
- 复活"死"异构系数

**代码位置**:
- 定期: `src/learners/k24_q_learner.py:_periodic_reset()`
- 自适应: `src/learners/k24_q_learner.py:_adaptive_reset()`

---

## 与原版对比

| 维度 | Kaleidoscope | K-2:4 |
|------|-------------|-------|
| **剪枝方式** | 非结构化 | 2:4 半结构化 |
| **硬件加速** | ❌ | ✅ ~2x |
| **可微性** | STE | Gumbel-Softmax |
| **激活统计** | 无 | EMA |
| **多样性损失** | L1 距离 | 点积 |
| **稀疏度** | 可变 | 固定 50% |

---

## 超参数配置

### 关键参数

```python
K24_args = {
    # 温度
    "temperature_init": 5.0,
    "temperature_min": 0.1,
    "anneal_start": 0.0,
    "anneal_end": 0.8,

    # EMA
    "ema_momentum": 0.99,

    # 异构性
    "hetero_init_scale": 0.01,

    # 多样性
    "div_coef": 0.1,
    "deque_len": 100,

    # 重置
    "reset_interval": 10000,
    "reset_ratio": 0.1,
}
```

**调优建议**:
- `div_coef`: 越高智能体差异越大,但可能降低性能
- `temperature_init`: 越高探索越强,收敛越慢
- `reset_ratio`: 0.1-0.2,过高可能不稳定

---

## 监控指标

### 训练过程

- **损失**: `loss_td`, `div_loss`, `div_coef`
- **温度**: `temperature`, `progress`
- **模式**: `pattern_*_prob`, `pattern_entropy`
- **稀疏度**: `sparsity_layer_*`, `overall_sparsity`
- **相似度**: `pattern_mean_similarity` (越低越好)

### TensorBoard

```bash
tensorboard --logdir=results/
```

---

## 性能预期

| 指标 | 预期值 |
|------|--------|
| **稀疏度** | ~50% (每个 1x4 组恰好 2 个非零) |
| **推理加速** | 1.5-2.0x (Ampere+ GPU) |
| **训练时间** | 0.9-1.0x (相比原版) |
| **最终性能** | 相当或略优 |
| **模式熵** | 1.5-2.5 (6 种模式较均匀分布) |

---

## 文档索引

1. **最终方案设计_K_24.md**: 算法理论设计
   - 核心模块详解
   - 数学推导
   - 预期效果

2. **K-2_4_Implementation_Documentation.md**: 实现细节
   - 代码与方案对应关系
   - 完整训练流程
   - 文件结构说明

3. **K-2_4_User_Guide.md**: 使用指南
   - 快速开始
   - 配置说明
   - 常见问题
   - 扩展定制

---

## 实现清单

### epymarl_kaleidoscope

- [x] `k24_pattern_gumbel_layer.py` - 模块 A + B
  - [x] EMAActivationTracker
  - [x] Pattern24Matrix
  - [x] SemiStructuredLinear24

- [x] `k24_diversity.py` - 模块 C
  - [x] PatternOrthogonalityLoss
  - [x] K24DiversityManager
  - [x] LayerPatternTracker

- [x] `k24_rnn_agent.py` - Agent
  - [x] K24_RNNAgent
  - [x] 温度退火
  - [x] 重置机制

- [x] `k24_q_learner.py` - Learner
  - [x] K24_QLearner
  - [x] 自适应多样性系数
  - [x] 日志记录

### Kalei_SMACv2

- [x] `layer/k24_pattern_gumbel_layer.py` (复制)
- [x] `layer/k24_diversity.py` (复制)
- [x] `k24_rnn_agent.py` - SMACv2 版本
- [x] `k24_nq_learner.py` - SMACv2 版本

---

## 引用

如果您使用本实现,请考虑引用:

```bibtex
@article{kaleidoscope2024,
  title={Kaleidoscope: Heterogeneous Agent Policies via Parameter Pruning},
  author={Everett, Richard and Lee, Jun and Sun, Minzhao},
  journal={arXiv preprint arXiv:2405.15820},
  year={2024}
}

@article{k24_2024,
  title={K-2:4: Hardware-Friendly Semi-Structured Sparsity for
         Heterogeneous Multi-Agent Reinforcement Learning},
  note={Implementation of Kaleidoscope with 2:4 sparsity},
  year={2024}
}
```

---

## 许可证

本实现遵循原 Kaleidoscope 项目的许可证。

---

## 贡献

欢迎提交 Issue 和 Pull Request!

### 贡献方向

- [ ] 支持更多稀疏模式 (4:8, 2:8)
- [ ] 添加更多多样性损失方法
- [ ] 优化 EMA 更新策略
- [ ] 支持分布式训练
- [ ] 添加更多环境测试

---

## 联系方式

- 项目地址: [GitHub 链接]
- 问题反馈: [Issues]
- 邮箱: [您的邮箱]

---

**最后更新**: 2024-01-04

**状态**: ✅ 实现完成,文档齐全
