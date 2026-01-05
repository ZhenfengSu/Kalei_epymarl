# K-2:4 配置文件总结

## 📋 配置文件清单

### ✅ epymarl_kaleidoscope 项目（3 个配置文件）

#### 1. `K24_qmix.yaml` - 标准配置
- **路径**: `src/config/algs/K24_qmix.yaml`
- **用途**: 平衡性能和多样性，适用于大多数场景
- **特点**:
  - `div_coef: 0.1` - 标准多样性
  - `temperature_init: 5.0` - 标准探索
  - `reset_interval: 10000` - 标准重置频率

#### 2. `K24_qmix_aggressive.yaml` - 激进多样性
- **路径**: `src/config/algs/K24_qmix_aggressive.yaml`
- **用途**: 强调智能体异构性，适合探索多样性上限
- **特点**:
  - `div_coef: 0.3` - 3x 强多样性
  - `temperature_init: 8.0` - 更强探索
  - `reset_interval: 5000` - 2x 更频繁重置
  - `reset_ratio: 0.2` - 2x 更多重置

#### 3. `K24_qmix_conservative.yaml` - 保守配置
- **路径**: `src/config/algs/K24_qmix_conservative.yaml`
- **用途**: 优先稳定性，适合简单环境
- **特点**:
  - `div_coef: 0.05` - 最小多样性
  - `temperature_init: 3.0` - 更快收敛
  - `reset_interval: 20000` - 2x 更少重置
  - `reset_ratio: 0.05` - 更少重置

---

### ✅ Kalei_SMACv2 项目（3 个配置文件）

#### 1. `K24_nq.yaml` - 标准配置
- **路径**: `src/src/config/algs/K24_nq.yaml`
- **用途**: 适用于中等复杂度的 SMACv2 地图
- **特点**:
  - `div_coef: 0.1` - 标准多样性
  - `temperature_init: 5.0` - 标准探索
  - `reset_interval: 10000` - 标准重置频率
  - 包含详细的 SMACv2 调优注释

#### 2. `K24_nq_small_map.yaml` - 小地图优化
- **路径**: `src/src/config/algs/K24_nq_small_map.yaml`
- **用途**: 优化小地图（3m, 2s3z）
- **特点**:
  - `div_coef: 0.05` - 低多样性（简单任务）
  - `temperature_init: 4.0` - 较低温度（快速收敛）
  - `anneal_end: 0.7` - 更早退火
  - `reset_interval: 20000` - 更少重置（稳定性）

#### 3. `K24_nq_large_map.yaml` - 大地图优化
- **路径**: `src/src/config/algs/K24_nq_large_map.yaml`
- **用途**: 优化大地图（corridor, 27m_vs_30m）
- **特点**:
  - `div_coef: 0.25` - 高多样性（复杂策略）
  - `temperature_init: 6.0` - 较高温度（更多探索）
  - `anneal_end: 0.9` - 更长退火期
  - `reset_interval: 5000` - 频繁重置（适应性强）
  - `use_adaptive_reset: True` - 启用自适应重置

---

## 🚀 使用方法

### epymarl_kaleidoscope

```bash
# 标准配置 - MPE 环境
python src/main.py \
    --config=K24_qmix \
    --env-config=mpe_simple_spread \
    --n-agents=3

# 标准配置 - SMAC 环境
python src/main.py \
    --config=K24_qmix \
    --env-config=smac \
    --env-args.map_name="3m"

# 激进多样性配置
python src/main.py \
    --config=K24_qmix_aggressive \
    --env-config=mpe_simple_reference \
    --n-agents=3

# 保守配置
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
    --K24_args.temperature_init=6.0
```

### Kalei_SMACv2

```bash
# 标准配置 - 中等地图
python src/main.py \
    --config=K24_nq \
    --env=smac_v2 \
    --map_name="3s5z"

# 小地图配置
python src/main.py \
    --config=K24_nq_small_map \
    --env=smac_v2 \
    --map_name="3m"

# 大地图配置
python src/main.py \
    --config=K24_nq_large_map \
    --env=smac_v2 \
    --map_name="corridor"

# 自定义参数
python src/main.py \
    --config=K24_nq \
    --env=smac_v2 \
    --map_name="6h_vs_8z" \
    --K24_args.div_coef=0.15 \
    --K24_args.use_adaptive_reset=True
```

---

## 📊 配置选择指南

### 按环境复杂度选择

| 环境类型 | 推荐配置 | div_coef | temp_init | reset_interval |
|---------|---------|----------|-----------|----------------|
| **MPE Simple** | K24_qmix_conservative | 0.05 | 3.0 | 20000 |
| **MPE Reference** | K24_qmix | 0.1 | 5.0 | 10000 |
| **SMAC 3m** | K24_nq_small_map | 0.05 | 4.0 | 20000 |
| **SMAC 2s3z** | K24_nq_small_map | 0.08 | 4.5 | 15000 |
| **SMAC 3s5z** | K24_nq | 0.1 | 5.0 | 10000 |
| **SMAC 6h_vs_8z** | K24_nq | 0.15 | 5.5 | 8000 |
| **SMAC Corridor** | K24_nq_large_map | 0.25 | 6.0 | 5000 |
| **SMAC 27m_vs_30m** | K24_nq_large_map | 0.3 | 7.0 | 5000 |

### 按需求选择

| 需求 | 推荐配置 | 原因 |
|------|---------|------|
| **快速验证** | K24_qmix_conservative / K24_nq_small_map | 快速收敛 |
| **最大异构性** | K24_qmix_aggressive / K24_nq_large_map | 强多样性 |
| **平衡性能** | K24_qmix / K24_nq | 标准配置 |
| **稳定训练** | K24_qmix_conservative / K24_nq_small_map | 低波动 |
| **探索上限** | K24_qmix_aggressive / K24_nq_large_map | 高探索 |

---

## 🎯 关键参数说明

### 1. div_coef（多样性系数）

```yaml
div_coef: 0.1    # 默认值
```

- **< 0.05**: 智能体策略趋于相同
- **0.05-0.15**: 推荐范围，平衡性能和多样性
- **0.15-0.3**: 高多样性，适合复杂环境
- **> 0.3**: 可能降低整体性能

### 2. temperature_init（初始温度）

```yaml
temperature_init: 5.0    # 默认值
```

- **3.0-4.0**: 快速收敛，适合简单环境
- **5.0-6.0**: 标准探索，适合大多数环境
- **7.0-10.0**: 强探索，适合复杂环境

### 3. reset_interval（重置间隔）

```yaml
reset_interval: 10000    # 默认值（步数）
```

- **5000-7000**: 频繁重置，动态环境
- **10000**: 标准重置，平衡
- **15000-20000**: 少重置，稳定环境

### 4. use_adaptive_reset（自适应重置）

```yaml
use_adaptive_reset: False    # 默认值
```

- **False**: 定期重置，稳定
- **True**: KL 散度触发重置，响应式（适合大地图）

---

## 📈 监控指标

训练时关注以下指标判断配置是否合适：

### ✅ 好的信号

```
- overall_sparsity: ~0.5          # 稀疏度约 50%
- pattern_mean_similarity: <0.3   # 智能体差异明显
- pattern_entropy: >1.5            # 模式分布较均匀
- temperature: 逐渐下降             # 正常退火
- div_coef: 0.05-0.5              # 自适应系数合理
```

### ⚠️ 需要调整的信号

```
# 如果看到这些，考虑提高 div_coef
- pattern_mean_similarity: >0.4   # 智能体太相似
- pattern_entropy: <1.0            # 模式分布不均

# 如果看到这些，考虑降低 div_coef
- loss_td: 剧烈波动                # 训练不稳定
- div_coef: >1.0                   # 自适应系数异常

# 如果看到这些，考虑调整温度
- 训练很慢，temperature 仍很高      # 退火太慢
- 收敛太快，性能不佳                # 退火太快
```

---

## 🔧 调优流程

### 第一步：从标准配置开始

```bash
python src/main.py --config=K24_qmix --env-config=smac --env-args.map_name="3m"
```

### 第二步：监控关键指标

```bash
# 查看 tensorboard 或日志
tensorboard --logdir=results/
```

### 第三步：根据表现调整

**如果智能体太相似**：
```yaml
K24_args:
  div_coef: 0.2              # 提高多样性
  reset_interval: 5000       # 更频繁重置
```

**如果训练不稳定**：
```yaml
K24_args:
  div_coef: 0.05             # 降低多样性
  reset_interval: 20000      # 减少重置
```

**如果收敛太慢**：
```yaml
K24_args:
  temperature_init: 3.0      # 降低初始温度
  anneal_end: 0.6            # 更早完成退火
```

---

## 📝 配置对比

### epymarl_kaleidoscope 三个配置对比

| 参数 | 标准配置 | 激进配置 | 保守配置 |
|------|---------|---------|---------|
| div_coef | 0.1 | 0.3 | 0.05 |
| temperature_init | 5.0 | 8.0 | 3.0 |
| anneal_end | 0.8 | 0.8 | 0.7 |
| reset_interval | 10000 | 5000 | 20000 |
| reset_ratio | 0.1 | 0.2 | 0.05 |

### Kalei_SMACv2 三个配置对比

| 参数 | 标准配置 | 小地图 | 大地图 |
|------|---------|--------|--------|
| div_coef | 0.1 | 0.05 | 0.25 |
| temperature_init | 5.0 | 4.0 | 6.0 |
| anneal_end | 0.8 | 0.7 | 0.9 |
| anneal_end_step | 800000 | 700000 | 900000 |
| reset_interval | 10000 | 20000 | 5000 |
| reset_ratio | 0.1 | 0.05 | 0.15 |
| use_adaptive_reset | False | False | **True** |

---

## ✅ 配置文件检查清单

使用配置前确认：

- [ ] 已选择合适的配置文件
- [ ] `hidden_dim` 是 4 的倍数（确保完美 2:4 稀疏）
- [ ] `t_max` 与 `anneal_end_step` 匹配
- [ ] 根据环境复杂度调整了 `div_coef`
- [ ] 根据环境动态性设置了 `reset_interval`
- [ ] 大地图考虑启用 `use_adaptive_reset`

---

## 🎓 学习资源

- **详细文档**: `K-2_4_Configuration_Guide.md`
- **实现文档**: `K-2_4_Implementation_Documentation.md`
- **用户指南**: `K-2_4_User_Guide.md`
- **方案设计**: `最终方案设计_K_24.md`

---

## 📞 问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 训练崩溃 | hidden_dim 不是 4 的倍数 | 改为 64, 128 等 |
| 智能体相同 | div_coef 太低 | 提高到 0.15-0.2 |
| 训练不稳定 | div_coef 太高或重置太频繁 | 降低到 0.05，提高 reset_interval |
| 收敛太慢 | temperature 太高 | 降低到 3.0-4.0 |
| 性能不佳 | 退火太快或重置太少 | 延长 anneal_end，降低 reset_interval |

---

**配置文件版本**: 1.0
**最后更新**: 2024-01-04
**状态**: ✅ 全部完成并测试
