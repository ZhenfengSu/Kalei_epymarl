# K-2:4 算法配置文件使用指南

## 配置文件列表

### epymarl_kaleidoscope 项目

| 配置文件 | 用途 | 特点 |
|---------|------|------|
| `K24_qmix.yaml` | 标准配置 | 平衡性能和多样性，适用于大多数场景 |
| `K24_qmix_aggressive.yaml` | 激进多样性 | 强调智能体异构性，适合探索多样性上限 |
| `K24_qmix_conservative.yaml` | 保守配置 | 优先稳定性，适合简单环境 |

### Kalei_SMACv2 项目

| 配置文件 | 用途 | 特点 |
|---------|------|------|
| `K24_nq.yaml` | 标准配置 | 适用于中等复杂度的 SMACv2 地图 |
| `K24_nq_small_map.yaml` | 小地图优化 | 快速收敛，适合 3m, 2s3z 等小地图 |
| `K24_nq_large_map.yaml` | 大地图优化 | 强调探索，适合 corridor, 27m_vs_30m 等大地图 |

---

## 配置参数详解

### 核心参数

#### 1. 温度退火 (Temperature Annealing)

```yaml
temperature_init: 5.0      # 初始温度（探索阶段）
temperature_min: 0.1       # 最小温度（收敛阶段）
anneal_start: 0.0          # 开始退火的训练进度（0.0-1.0）
anneal_end: 0.8            # 完成退火的训练进度（0.0-1.0）
anneal_end_step: 800000    # 退火结束步数（0.8 * t_max）
```

**调优建议**：
- **高温度** (5.0-10.0)：更多探索，收敛较慢，适合复杂环境
- **低温度** (3.0-5.0)：更快收敛，适合简单环境
- **退火时长**：应覆盖训练的 70-90%

**环境对应**：
```yaml
# 简单环境 (MPE, 小地图)
temperature_init: 3.0-4.0
anneal_end: 0.7

# 中等环境 (标准 SMAC)
temperature_init: 5.0
anneal_end: 0.8

# 复杂环境 (大地图, 高动态)
temperature_init: 6.0-8.0
anneal_end: 0.85-0.9
```

---

#### 2. 多样性系数 (Diversity Coefficient)

```yaml
div_coef: 0.1              # 基础多样性系数
deque_len: 100             # 损失历史队列长度（用于自适应系数）
```

**调优建议**：
- **太低** (< 0.05)：智能体策略趋于相同，失去异构性
- **太高** (> 0.5)：可能降低整体性能
- **自适应机制**：会根据 `td_loss / div_loss` 比例自动调整

**环境对应**：
```yaml
# 同质智能体（相同类型，相同角色）
div_coef: 0.05-0.1

# 异质智能体（不同类型，不同角色）
div_coef: 0.1-0.2

# 高度异质（需要显著差异化）
div_coef: 0.2-0.3
```

---

#### 3. 自适应重置 (Adaptive Resetting)

```yaml
reset_interval: 10000      # 定期重置间隔（环境步数）
reset_ratio: 0.1           # 重置比例（0.05-0.2，即 5%-20%）
use_adaptive_reset: False  # 是否启用 KL 散度自适应重置
kl_threshold: 0.1          # KL 散度阈值
```

**调优建议**：
- **定期重置**：
  - 稳定环境：15000-20000 步
  - 动态环境：5000-10000 步

- **重置比例**：
  - 0.05（5%）：保守，适合稳定训练
  - 0.1（10%）：标准，平衡探索和稳定
  - 0.2（20%）：激进，适合需要大量探索的场景

- **自适应重置**：
  - 优点：响应策略变化，更智能
  - 缺点：可能不稳定
  - 推荐：大地图或高动态环境启用

---

#### 4. EMA 动量 (EMA Momentum)

```yaml
ema_momentum: 0.99         # EMA 系数（0.9-0.999）
```

**调优建议**：
- **0.99**：推荐默认值，适用于大多数场景
- **0.999**：非常稳定的环境，更平滑的统计
- **0.95**：高度动态的环境，更快适应

---

#### 5. 异构系数初始化 (Heterogeneity Initialization)

```yaml
hetero_init_scale: 0.01    # α 系数初始化标准差
```

**调优建议**：
- **0.005**：小初始化，更保守的探索
- **0.01**：标准初始化，平衡探索
- **0.02**：大初始化，更强异构性

---

## 使用示例

### epymarl_kaleidoscope

#### 基础使用

```bash
# 使用标准配置训练 MPE 环境
python src/main.py \
    --config=K24_qmix \
    --env-config=mpe_simple_spread \
    --n-agents=3

# 使用标准配置训练 SMAC 环境
python src/main.py \
    --config=K24_qmix \
    --env-config=smac \
    --env-args.map_name="3m"
```

#### 高级使用

```bash
# 使用激进多样性配置
python src/main.py \
    --config=K24_qmix_aggressive \
    --env-config=mpe_simple_spread \
    --n-agents=3

# 使用保守配置
python src/main.py \
    --config=K24_qmix_conservative \
    --env-config=smac \
    --env-args.map_name="2s3z"

# 自定义参数
python src/main.py \
    --config=K24_qmix \
    --env-config=smac \
    --env-args.map_name="3m" \
    --K24_args.div_coef=0.2 \
    --K24_args.temperature_init=6.0 \
    --K24_args.reset_interval=5000
```

#### 超参数搜索

```bash
# 创建搜索配置文件
cat > search_k24.yaml << 'EOF'
grid-search:
  "--config":
    - "K24_qmix"

  K24_args.div_coef:
    - 0.05
    - 0.1
    - 0.2

  K24_args.temperature_init:
    - 3.0
    - 5.0
    - 7.0

  K24_args.reset_interval:
    - 5000
    - 10000
    - 20000

grid-search-groups:
  env0:
    - "--env-config": "smac"
    - env_args.map_name: "3m"
EOF

# 运行搜索
python src/search.py --search-config=search_k24.yaml
```

---

### Kalei_SMACv2

#### 基础使用

```bash
# 使用标准配置（中等地图）
python src/main.py \
    --config=K24_nq \
    --env=smac_v2 \
    --map_name="3s5z"

# 使用小地图配置
python src/main.py \
    --config=K24_nq_small_map \
    --env=smac_v2 \
    --map_name="3m"

# 使用大地图配置
python src/main.py \
    --config=K24_nq_large_map \
    --env=smac_v2 \
    --map_name="corridor"
```

#### 高级使用

```bash
# 自定义地图参数
python src/main.py \
    --config=K24_nq \
    --env=smac_v2 \
    --map_name="6h_vs_8z" \
    --K24_args.div_coef=0.15 \
    --K24_args.temperature_init=5.5 \
    --K24_args.use_adaptive_reset=True

# 针对特定地图调优
python src/main.py \
    --config=K24_nq \
    --env=smac_v2 \
    --map_name="2s3z" \
    --K24_args.div_coef=0.08 \
    --K24_args.reset_interval=15000 \
    --K24_args.anneal_end_step=600000
```

---

## 配置对比表

### 不同场景推荐配置

| 场景 | 推荐配置 | div_coef | temp_init | reset_interval | use_adaptive |
|------|---------|----------|-----------|----------------|--------------|
| **MPE Simple Spread** | K24_qmix_conservative | 0.05 | 3.0 | 20000 | False |
| **MPE Reference** | K24_qmix | 0.1 | 5.0 | 10000 | False |
| **SMAC 3m** | K24_nq_small_map | 0.05 | 4.0 | 20000 | False |
| **SMAC 2s3z** | K24_nq_small_map | 0.08 | 4.5 | 15000 | False |
| **SMAC 3s5z** | K24_nq | 0.1 | 5.0 | 10000 | False |
| **SMAC 6h_vs_8z** | K24_nq | 0.15 | 5.5 | 8000 | False |
| **SMAC Corridor** | K24_nq_large_map | 0.25 | 6.0 | 5000 | True |
| **SMAC 27m_vs_30m** | K24_nq_large_map | 0.3 | 7.0 | 5000 | True |

---

## 配置调试技巧

### 1. 诊断工具

训练时监控以下指标：

```python
# 在 tensorboard 或日志中查看
# 温度相关
- temperature                    # 应从 5.0 降至 0.1
- progress                       # 训练进度

# 多样性相关
- div_loss                       # 多样性损失（越低越好）
- div_coef                       # 自适应系数（会自动调整）
- pattern_mean_similarity        # 模式相似度（应 < 0.3）
- pattern_entropy                # 模式熵（应 > 1.5）

# 稀疏度相关
- overall_sparsity               # 整体稀疏度（应 ~0.5）
- sparsity_layer_*               # 各层稀疏度（应 ~0.5）
```

### 2. 常见问题诊断

#### 问题 1：智能体策略趋于相同

**症状**：
- `pattern_mean_similarity` > 0.4
- `pattern_entropy` < 1.0

**解决方案**：
```yaml
K24_args:
  div_coef: 0.2              # 提高多样性系数
  reset_interval: 5000       # 更频繁重置
  reset_ratio: 0.15          # 重置更多系数
```

#### 问题 2：训练不稳定

**症状**：
- `loss_td` 剧烈波动
- `div_coef` 异常高 (> 1.0)

**解决方案**：
```yaml
K24_args:
  div_coef: 0.05             # 降低多样性系数
  reset_interval: 20000      # 减少重置频率
  reset_ratio: 0.05          # 减少重置比例
  use_adaptive_reset: False  # 禁用自适应重置
```

#### 问题 3：收敛过慢

**症状**：
- 训练很长时间性能仍不佳
- `temperature` 仍很高

**解决方案**：
```yaml
K24_args:
  temperature_init: 3.0      # 降低初始温度
  anneal_end: 0.6            # 更早完成退火
  anneal_end_step: 600000    # 更短的退火期
```

#### 问题 4：稀疏度不是 50%

**症状**：
- `overall_sparsity` 不是 ~0.5

**可能原因**：
- 权重维度不是 4 的倍数

**解决方案**：
```yaml
# 确保 hidden_dim 是 4 的倍数
hidden_dim: 64               # ✓ 正确
hidden_dim: 65               # ✗ 可能导致问题
```

---

## 配置迁移指南

### 从原版 Kaleidoscope 迁移

```yaml
# 原 Kaleidoscope 配置
Kalei_args:
  div_coef: 0.5
  threshold_init_scale: 5.0
  threshold_init_bias: 5.0
  reset_interval: 200000
  reset_ratio: 0.1

# K-2:4 等效配置
K24_args:
  div_coef: 0.1              # 降低 5x（新的损失更直接）
  temperature_init: 5.0      # 新增参数
  anneal_end_step: 800000    # 新增退火控制
  reset_interval: 200000     # 保持相同
  reset_ratio: 0.1           # 保持相同
```

**注意**：
- K-2:4 的多样性损失更直接，通常需要更小的 `div_coef`
- 添加了温度退火机制
- 异构系数从 `threshold` 改为 `hetero_alpha`

---

## 完整配置示例

### 极简配置（最小示例）

```yaml
# K24_qmix_minimal.yaml
agent: "k24_rnn"
learner: "k24_q_learner"
mac: "kalei_mac"
mixer: "qmix"
use_rnn: True

K24_args:
  temperature_init: 5.0
  temperature_min: 0.1
  div_coef: 0.1
  reset_interval: 10000
```

### 生产配置（推荐）

```yaml
# K24_qmix_production.yaml
agent: "k24_rnn"
learner: "k24_q_learner"
mac: "kalei_mac"
mixer: "qmix"
use_rnn: True

K24_args:
  # 温度退火
  temperature_init: 5.0
  temperature_min: 0.1
  anneal_start: 0.0
  anneal_end: 0.8
  anneal_end_step: 800000

  # EMA 和异构性
  ema_momentum: 0.99
  hetero_init_scale: 0.01

  # 多样性损失
  div_coef: 0.1
  deque_len: 100

  # 自适应重置
  reset_interval: 10000
  reset_ratio: 0.1
  use_adaptive_reset: False
  kl_threshold: 0.1

# 其他标准参数
buffer_size: 5000
batch_size: 32
lr: 0.0005
gamma: 0.99
```

---

## 总结

### 快速选择指南

1. **不确定使用哪个？** → 使用 `K24_qmix.yaml` 或 `K24_nq.yaml`（标准配置）
2. **简单环境？** → 使用 `_conservative` 或 `_small_map` 配置
3. **复杂环境？** → 使用 `_aggressive` 或 `_large_map` 配置
4. **想要更多异构性？** → 提高 `div_coef` 到 0.2-0.3
5. **训练不稳定？** → 降低 `div_coef` 到 0.05，提高 `reset_interval`

### 配置最佳实践

- ✅ 从标准配置开始
- ✅ 根据环境复杂度调整 `div_coef` 和 `temperature_init`
- ✅ 监控 `pattern_mean_similarity` 和 `overall_sparsity`
- ✅ 小幅度调参（每次调整 1-2 个参数）
- ❌ 不要同时调整多个参数
- ❌ 不要将 `div_coef` 设置得过高（> 0.5）

---

**最后更新**: 2024-01-04
