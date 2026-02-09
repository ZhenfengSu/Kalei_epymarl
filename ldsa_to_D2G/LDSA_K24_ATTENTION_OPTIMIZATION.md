# LDSA with TAAE-Style Attention and K24 2:4 Sparsity

## Overview

This document describes the optimization implementation for LDSA (Learning Dynamic Subtask Assignment) that integrates:

1. **TAAE-style Attention Mechanism**: Multi-head attention that integrates role embeddings, global state, and local observations
2. **K24 2:4 Semi-Structured Sparsity**: Pattern-based sparse masks for efficient inference and agent heterogeneity

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Observation  │  │ Global State │  │   Hidden States      │  │
│  │  (Local)     │  │ (Environment)│  │   (Previous)         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼──────────────────┼─────────────────────┼──────────────┘
          │                  │                     │
          ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Embedding Network                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  K24SparseLinear (fc1) → GRU → K24SparseLinear (fc2)   │   │
│  │  [batch*agents, obs] → [batch*agents, embed_dim]        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │       OptimizedDynamicSubtaskGenerator                  │   │
│  │  - Aggregates agent embeddings                          │   │
│  │  - Fuses with global state (K24 sparse if enabled)      │   │
│  │  - Predicts optimal subtask count                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Attention Module                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MultiAgentAttentionModule                              │   │
│  │  ┌────────────────────────────────────────────────┐    │   │
│  │  │  GlobalStateFusion (GIN-like)                   │    │   │
│  │  │  - Observation → weights for state processing   │    │   │
│  │  │  - State * weights(obs) + obs → z_fused         │    │   │
│  │  └────────────────────────────────────────────────┘    │   │
│  │                         │                                │   │
│  │                         ▼                                │   │
│  │  ┌────────────────────────────────────────────────┐    │   │
│  │  │  RoleStateAttention (Multi-Head)                │    │   │
│  │  │  - Query: Role/subtask embedding                │    │   │
│  │  │  - Key/Value: Fused state-observation            │    │   │
│  │  │  - Output: Context vector for each agent         │    │   │
│  │  └────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Subtask Selection                            │
│  - Enhanced role embeddings (from attention)                    │
│  - Subtask embeddings (learnable)                               │
│  - Attention-based subtask selection                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Subtask Policy Network                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  K24SparseLinear (fc1) → GRU → K24SparseLinear (fc2)   │   │
│  │  → Q-values for each subtask-action pair                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         Q-values
```

## Key Components

### 1. Global State Fusion (TAAE-style)

**Location**: `src/modules/attention/role_state_attention.py::GlobalStateFusion`

This module implements the "Coach Network" (GIN) from TAAE, which fuses global state with local observations using a hypernetwork approach.

**Key Features**:
- Observation-conditioned weight generation for state processing
- Personalized fusion per agent
- Enables centralized training with ground-truth global state

**Mathematical Formulation**:
```
z_state = (W_1(obs) @ state) + b_1(obs)
z_obs = Encoder(obs)
z_fused = z_state + z_obs
```

### 2. Role-State-Observation Attention

**Location**: `src/modules/attention/role_state_attention.py::RoleStateAttention`

Multi-head attention mechanism that extracts role-specific information from the fused state-observation representation.

**Key Features**:
- Query: Derived from role/subtask embeddings
- Key/Value: Derived from fused state-observation
- Scaled dot-product attention with multiple heads
- Layer normalization and residual connections

**Mathematical Formulation**:
```
Q = W_Q @ role_emb
K = W_K @ z_fused
V = W_V @ z_fused
attention = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### 3. K24 2:4 Semi-Structured Sparse Linear Layer

**Location**: `src/modules/layer/k24_sparse_linear.py::SemiStructuredLinear24`

Implements 2:4 semi-structured sparsity using the K-2:4 algorithm with three modules:

#### Module A: Dynamic Heterogeneous Scoring

```
S_i = |W_shared| * EMA(|A_i|) * sigmoid(alpha_i)
```

- `W_shared`: Shared weights across all agents
- `EMA(|A_i|)`: Exponential moving average of activation magnitudes
- `alpha_i`: Agent-specific heterogeneity coefficients

#### Module B: Pattern-based Projection with Gumbel-Softmax

Six valid 2:4 sparse patterns:
- Pattern 0: `[1, 1, 0, 0]`
- Pattern 1: `[1, 0, 1, 0]`
- Pattern 2: `[1, 0, 0, 1]`
- Pattern 3: `[0, 1, 1, 0]`
- Pattern 4: `[0, 1, 0, 1]`
- Pattern 5: `[0, 0, 1, 1]`

**Pattern Selection**:
```
# Forward: discrete selection
pattern_probs = gumbel_softmax(scores @ pattern_matrix^T, tau=temperature, hard=True)
mask = pattern_probs @ pattern_matrix

# Backward: continuous gradient
# Gradient flows through soft probabilities before discretization
```

#### Module C: Pattern Orthogonality Diversity Loss

**Location**: `src/modules/layer/k24_diversity.py::PatternOrthogonalityLoss`

Encourages different agents to select different sparse patterns:

```
L_div = mean(pi_a * pi_b) for all agent pairs (a, b)
```

### 4. Dynamic Subtask Generation with Global State

**Location**: `src/modules/agents/ldsa_k24_agent.py::OptimizedDynamicSubtaskGenerator`

Enhanced version of the original LDSA subtask generator that:

- Uses global state as additional context
- Optionally applies K24 sparsity to the fusion layer
- Implements caching for efficient inference

## Usage

### Training

To train LDSA with K24 sparsity and attention:

```bash
# For SMAC environments (e.g., corridor)
python src/main.py --config=algs/ldsa_k24 --env-config=envs/sc2

# For MPE environments
python src/main.py --config=algs/ldsa_k24 --env-config=envs/mpe_simple_spread
```

### Configuration Parameters

#### K24 Sparsity Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_k24` | `True` | Enable K24 2:4 sparsity |
| `K24_temperature_init` | `5.0` | Initial Gumbel-Softmax temperature |
| `K24_temperature_min` | `0.1` | Minimum temperature |
| `K24_anneal_end_step` | `4000000` | Step when annealing completes |
| `K24_div_coef` | `0.1` | Diversity loss coefficient |
| `K24_reset_interval` | `500000` | Heterogeneity coefficient reset interval |
| `K24_finetune_start_ratio` | `0.8` | Start finetuning at 80% of training |

#### Attention Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_attention` | `True` | Enable attention mechanism |
| `K24_use_global_state` | `True` | Use global state in attention |
| `attention_z_dims` | `64` | Latent dimension for state-obs fusion |
| `attention_num_heads` | `4` | Number of attention heads |
| `attention_hidden_dim` | `64` | Hidden dimension for projections |

### Training Schedule

The training follows a two-phase schedule:

**Phase 1: Exploration (0-80% of training)**
- Temperature annealing: 5.0 → 0.1
- Diversity loss: Active with adaptive coefficient
- Mask resets: Every 500K steps
- Pattern exploration: High

**Phase 2: Finetuning (80-100% of training)**
- Masks: Frozen
- Learning rate: Decayed by 10%
- Diversity loss: Disabled
- Focus: Weight optimization with fixed structure

## Implementation Details

### File Structure

```
src/modules/
├── attention/
│   ├── __init__.py
│   └── role_state_attention.py    # TAAE-style attention
├── layer/
│   ├── __init__.py
│   ├── k24_sparse_linear.py       # K24 sparse layer
│   └── k24_diversity.py           # Diversity loss
└── agents/
    ├── __init__.py
    ├── ldsa_agent.py              # Original LDSA agent
    ├── ldsa_agent_optim.py        # Optimized LDSA agent
    └── ldsa_k24_agent.py          # NEW: LDSA + K24 + Attention

src/learners/
├── __init__.py
├── ldsa_learner.py                # Original LDSA learner
└── ldsa_k24_learner.py            # NEW: LDSA + K24 learner

src/config/
├── algs/
│   ├── ldsa.yaml                  # Original LDSA config
│   └── ldsa_k24.yaml              # NEW: LDSA + K24 config
└── envs/
    ├── sc2.yaml                   # SC2 environment
    └── mpe_simple_spread.yaml     # MPE environment
```

### Key Classes

1. **`K24SparseLDSAAgent`**: Main agent integrating all components
2. **`LDSAK24Learner`**: Learner with K24-specific training logic
3. **`MultiAgentAttentionModule`**: Complete attention pipeline
4. **`SemiStructuredLinear24`**: K24 sparse linear layer

## Optimization Benefits

### 1. Computational Efficiency

- **2:4 Sparsity**: 50% reduction in FLOPs with structured patterns
- **Hardware Acceleration**: Compatible with NVIDIA Ampere 2:4 sparse tensor cores
- **Memory Efficiency**: Sparse activation representation

### 2. Agent Heterogeneity

- **Agent-specific masks**: Each agent learns different sparse patterns
- **Diversity loss**: Encourages pattern differentiation
- **Specialization**: Agents specialize in different subtasks

### 3. Global Information Integration

- **Attention mechanism**: Efficiently extracts relevant global information
- **Personalized fusion**: Each agent processes global state differently
- **Centralized training**: Uses ground-truth state for better learning

### 4. Training Stability

- **EMA activation tracking**: Handles RL distribution shift
- **Temperature annealing**: Smooth transition from exploration to exploitation
- **Adaptive diversity coefficient**: Balances TD loss and diversity loss

## Comparison with Baseline LDSA

| Feature | Baseline LDSA | LDSA + K24 + Attention |
|---------|---------------|------------------------|
| Global State | Not used | Used via attention |
| Sparsity | None | 50% (2:4 structured) |
| Agent Heterogeneity | Role embeddings only | Role + sparse patterns |
| Attention | None | Multi-head attention |
| Hardware Acceleration | No | Yes (2:4 sparse) |
| Training Complexity | Standard | Moderate (extra losses) |
| Inference Speed | Standard | ~2x faster (theoretical) |

## Troubleshooting

### Issue: Training instability

**Solution**: Increase `K24_div_coef` to encourage more pattern diversity, or decrease `K24_temperature_init` for less exploration.

### Issue: Poor convergence

**Solution**: Check if `K24_anneal_end_step` is appropriate for your `t_max`. It should be around 80% of total training steps.

### Issue: CUDA out of memory

**Solution**: Reduce `batch_size` or disable attention (`use_attention: False`) to save memory.

### Issue: Sparsity not ~50%

**Solution**: Check that input dimensions are divisible by 4 for 2:4 pattern grouping. The implementation includes padding if needed.

## References

1. **TAAE (Team-Aware Attention Extraction)**: `/mnt/lc_gpu_test/marl_final_exp/ptde_ldsa_attn`
   - Global state fusion module
   - Multi-head attention for agent coordination

2. **Kaleidoscope K24**: `/mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/Kaleidoscope/Kalei_SMACv2`
   - 2:4 semi-structured sparsity implementation
   - Pattern-based Gumbel-Softmax
   - Dynamic heterogeneous scoring

3. **Design Documents**:
   - `/mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/最终方案设计_K_24.md`
   - `/mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/Kaleidoscope/Kalei_SMACv2/SMACv2_K24_IMPLEMENTATION_SUMMARY.md`
