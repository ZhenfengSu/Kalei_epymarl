# K-2:4 (K24_nq) Algorithm Implementation for MPE

## Summary

Successfully implemented the K-2:4 algorithm with semi-structured sparsity for Multi-Agent Particle Environments (MPE) in the epymarl_kaleidoscope repository.

## Implementation Date

January 11, 2025

## Files Created

### 1. Core Layer Modules (`src/modules/layer/`)

#### `k24_pattern_gumbel_layer.py` (396 lines)
- **EMAActivationTracker**: Exponential moving average tracker for activation magnitudes
- **Pattern24Matrix**: Defines the 6 valid 2:4 sparse patterns
- **SemiStructuredLinear24**: Custom Linear layer with 2:4 semi-structured sparsity
  - Module A: Dynamic Heterogeneous Scoring
  - Module B: Pattern-based Projection with Gumbel-Softmax
  - Temperature annealing
  - Adaptive resetting mechanism

#### `k24_diversity.py` (250 lines)
- **PatternOrthogonalityLoss**: Module C - Pattern orthogonality diversity loss
- **K24DiversityManager**: Manages diversity loss with adaptive coefficient
- **LayerPatternTracker**: Helper class to track pattern probabilities

#### `__init__.py` (17 lines)
- Registers all layer components

### 2. Agent Implementation (`src/modules/agents/`)

#### `k24_rnn_agent.py` (210 lines)
- **K24_RNNAgent_1R3**: RNN agent with 1R3 architecture (1 recurrent + 3 hidden layers)
  - Uses n_agents (not n_unit_types like SMACv2)
  - Agent-specific 2:4 semi-structured sparse masks
  - Pattern tracking and sparsity statistics
  - Temperature annealing
  - Heterogeneity coefficient resetting

### 3. Learner Implementation (`src/learners/`)

#### `k24_q_learner.py` (207 lines)
- **K24_QLearner**: Extends QLearner with K-2:4 features
  - TD loss + Pattern orthogonality diversity loss
  - Adaptive diversity coefficient
  - Temperature annealing
  - Periodic mask resetting
  - Comprehensive logging (sparsity, patterns, diversity metrics)

### 4. Configuration File

#### `config/algs/K24_nq.yaml`
- Complete K-2:4 configuration for MPE environments
- Tuned parameters for MPE characteristics
- Environment-specific tuning notes

### 5. Updated Registration Files

- `src/modules/agents/__init__.py`: Added K24_RNNAgent_1R3 registration
- `src/learners/__init__.py`: Added K24_QLearner registration

## Key Features Implemented

### Module A: Dynamic Heterogeneous Scoring
- EMA activation tracking (momentum=0.99)
- Shared weights with agent-specific modulation
- Learnable heterogeneity coefficients (α)

### Module B: Pattern-based Projection
- 6 valid 2:4 sparse patterns
- Pattern projection matrix
- Gumbel-Softmax for differentiable sampling
- Strict 2:4 constraint (50% sparsity)

### Module C: Pattern Orthogonality
- Probability distribution comparison
- Minimizes dot product between pattern distributions
- Adaptive diversity coefficient

### Training Dynamics
- Temperature annealing (5.0 → 0.1)
- Periodic mask resetting (every 10K steps)
- Comprehensive logging and statistics

## Usage Example

```python
# Configuration
args = types.SimpleNamespace(
    n_agents=3,
    hidden_dim=64,
    n_actions=6,
    use_rnn=True,
    t_max=500000,
    K24_args={
        "temperature_init": 5.0,
        "temperature_min": 0.1,
        "anneal_end_step": 400000,
        "div_coef": 0.1,
        "reset_interval": 10000,
    }
)

# Create agent and learner
agent = K24_RNNAgent(input_shape=128, args=args)
learner = K24_QLearner(mac, scheme, logger, args)

# Training loop
for episode in range(max_episodes):
    batch = collect_experience()
    loss = learner.train(batch, t_env, episode_num)
```

## Differences from SMACv2 Implementation

1. **Agent vs Unit Types**: Uses `n_agents` instead of `n_unit_types`
2. **Simpler Environment**: MPE is less complex than SMACv2
3. **Faster Annealing**: Can use faster temperature schedules
4. **Stable Dynamics**: More stable state distributions in MPE

## Verified Components

All components are successfully registered:
- Agent: `k24_rnn_1R3` ✓
- Learner: `k24_q_learner` ✓

## Testing

Run training with:
```bash
python3 src/main.py --config=algs/K24_nq --env-config=envs/simple_spread
```

## Correspondence to Design Document

The implementation follows the design in `/mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/最终方案设计_K_24.md`:

- **Module A**: Implemented in `SemiStructuredLinear24._compute_heterogeneous_scores()`
- **Module B**: Implemented in `SemiStructuredLinear24._pattern_gumbel_softmax()`
- **Module C**: Implemented in `PatternOrthogonalityLoss.forward()`
- **Training Dynamics**: Implemented in `K24_QLearner.train()`

## Next Steps

1. Test training on simple MPE environments (simple_spread, simple_reference)
2. Compare performance against baseline (QMix, Kalei)
3. Tune hyperparameters for specific MPE tasks
4. Benchmark sparsity and computational efficiency

## Notes

- Strict 2:4 sparsity ensures hardware acceleration compatibility
- Pattern-based approach maintains differentiability
- Adaptive mechanisms handle RL distribution shift
- Diversity loss promotes agent specialization




# 二次更新
  修改内容

  1. 配置文件 (K24_nq.yaml)
  temperature_init: 5.0           # Initial temperature
  temperature_min: 0.2            # Minimum temperature
  anneal_start: 0.0               # Start annealing at training start
  anneal_end: 0.8                 # Complete annealing at 80% progress
  anneal_end_step: 2400000         # End step (0.8 * t_max for 5000K timesteps)
  2. 在gumble softmax采样中使用更稳定的logits计算方式，通过添加layer normalization来稳定训练过程。


# 三次更新：Rewind & Finetune（彩票假设）

  修改内容

  1. 配置文件 (K24_nq.yaml)

  新增微调参数：
  # ========== Rewind & Finetune (Lottery Ticket Hypothesis) ==========
  finetune_start_ratio: 0.8       # 在训练80%时开始微调（最后20%）
  finetune_lr_decay: 0.1          # 学习率衰减至10%

  2. Layer模块 (k24_pattern_gumbel_layer.py)

  - 新增标志和缓冲区：
    - mask_frozen: 布尔标志，跟踪mask是否冻结
    - frozen_mask: 存储冻结的mask [n_agents, out_features, in_features]

  - 新增方法：
    - freeze_mask(): 冻结当前所有智能体的mask模式
      - 对所有智能体计算当前mask
      - 存储到frozen_mask缓冲区
      - 设置mask_frozen标志为True
    - unfreeze_mask(): 解冻mask（恢复训练）
    - is_mask_frozen(): 查询mask是否冻结

  - forward方法修改：
    - 当mask_frozen=False：正常使用Gumbel-Softmax采样
    - 当mask_frozen=True：跳过EMA更新和Gumbel-Softmax，直接使用frozen_mask
    - 设置last_pattern_probs=None以跳过多样性损失

  3. Learner模块 (k24_q_learner.py)

  - 新增参数：
    - finetune_start_ratio: 开始微调的训练比例（默认0.8，即80%）
    - finetune_lr_decay: 微调时学习率衰减因子（默认0.1）
    - finetune_started: 标志是否已进入微调阶段
    - base_lr: 保存原始学习率
    - finetune_start_step: 计算得出的开始微调的步数

  - 新增方法 _start_finetune(t_env)：
    1. 冻结所有层的mask（停止Gumbel-Softmax采样）
    2. 降低学习率（默认至10%）
    3. 禁用异构系数alpha的梯度（只微调权重W）
    4. 记录转换日志

  - train方法修改：
    - 检查是否到达微调时间点
    - 到达时调用_start_finetune()
    - 微调模式下跳过周期性mask重置
    - 微调模式下跳过多样性损失计算
    - 新增日志：finetune_mode

  训练时间线

  0% ──────────────────────────────────────────────────────────────────── 100%
      |                 |                    |
      训练初期          开始微调点           训练结束
      (mask可变)        (冻结mask)           (固定mask微调W)

      finetune_start_ratio=0.8

  微调阶段行为：
  - Mask完全固定（不进行Gumbel采样）
  - EMA不再更新
  - 异构系数α不再优化
  - 只有共享权重W进行微调
  - 学习率降低至10%

  预期效果

  根据彩票假设（Lottery Ticket Hypothesis）：
  - 存活权重能够稳定下来，无需适应mask的变化
  - 专门针对最终结构进行精细调整
  - 通常能带来2%-5%的性能提升

  使用方式

  配置文件中的参数可根据任务调整：
  - finetune_start_ratio: 调整开始微调的时间点（0.75-0.85推荐）
  - finetune_lr_decay: 微调学习率衰减（0.05-0.2推荐）
