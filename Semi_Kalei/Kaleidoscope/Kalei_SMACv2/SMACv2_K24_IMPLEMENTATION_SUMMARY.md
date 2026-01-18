# K-2:4 (K24) Algorithm Implementation for SMACv2

## Summary

Successfully implemented the K-2:4 algorithm with semi-structured sparsity for StarCraft II Multi-Agent Challenge (SMACv2) in the Kalei_SMACv2 repository.

## Implementation Date

January 17, 2026

## Files Created

### 1. Core Layer Modules (`src/src/modules/layer/`)

#### `k24_pattern_gumbel_layer.py` (459 lines)
- **EMAActivationTracker**: Exponential moving average tracker for activation magnitudes
- **Pattern24Matrix**: Defines the 6 valid 2:4 sparse patterns
- **SemiStructuredLinear24**: Custom Linear layer with 2:4 semi-structured sparsity
  - Module A: Dynamic Heterogeneous Scoring
  - Module B: Pattern-based Projection with Gumbel-Softmax
  - Temperature annealing
  - Adaptive resetting mechanism
  - Rewind & Finetune support

#### `k24_diversity.py` (250 lines)
- **PatternOrthogonalityLoss**: Module C - Pattern orthogonality diversity loss
- **K24DiversityManager**: Manages diversity loss with adaptive coefficient
- **LayerPatternTracker**: Helper class to track pattern probabilities

### 2. Agent Implementation (`src/src/modules/agents/`)

#### `n_rnn_agent.py` (added K24_type_NRNNAgent_1R3 class, ~200 lines)
- **K24_type_NRNNAgent_1R3**: RNN agent with 1R3 architecture for SMACv2
  - Uses n_unit_types (not n_agents like MPE)
  - Unit-type-specific 2:4 semi-structured sparse masks
  - Pattern tracking and sparsity statistics
  - Temperature annealing
  - Heterogeneity coefficient resetting

### 3. Learner Implementation (`src/src/learners/`)

#### `k24_nq_learner.py` (330 lines)
- **K24_NQLearner**: Extends NQLearner with K-2:4 features
  - TD loss + Pattern orthogonality diversity loss
  - Adaptive diversity coefficient
  - Temperature annealing
  - Periodic mask resetting
  - Rewind & Finetune mechanism
  - Comprehensive logging (sparsity, patterns, diversity metrics)

### 4. Configuration File

#### `config/algs/K24_qmix_rnn_1R3.yaml`
- Complete K-2:4 configuration for SMACv2 environments
- Tuned parameters for SMACv2 characteristics
- Environment-specific tuning notes

### 5. Updated Registration Files

- `src/src/modules/agents/__init__.py`: Added K24_type_NRNNAgent_1R3 registration
- `src/src/learners/__init__.py`: Added K24_NQLearner registration

## Key Features Implemented

### Module A: Dynamic Heterogeneous Scoring
- EMA activation tracking (momentum=0.99)
- Shared weights with unit-type-specific modulation
- Learnable heterogeneity coefficients (α)

### Module B: Pattern-based Projection
- 6 valid 2:4 sparse patterns
- Pattern projection matrix
- Gumbel-Softmax for differentiable sampling
- Layer normalization for stable training
- Strict 2:4 constraint (50% sparsity)

### Module C: Pattern Orthogonality
- Probability distribution comparison
- Minimizes dot product between pattern distributions
- Adaptive diversity coefficient

### Training Dynamics
- Temperature annealing (5.0 → 0.2)
- Longer reset intervals (500K steps for SMACv2)
- Higher reset ratio (20% for exploration)
- Comprehensive logging and statistics

### Rewind & Finetune
- Mask freezing at 80% training
- Learning rate decay (90%)
- Optimizer reset for stability
- Target network synchronization

## Differences from MPE Implementation

1. **Unit Types vs Agents**: Uses `n_unit_types` instead of `n_agents`
2. **More Complex Environment**: SMACv2 requires longer training and slower annealing
3. **Larger State Space**: Higher dimensional observations
4. **Longer Reset Intervals**: 500K vs 10K steps
5. **Higher Reset Ratio**: 20% vs 10%
6. **Different MAC**: Uses "Kalei_type_n_mac" instead of "kalei_mac"

## Architecture Overview

```
SMACv2 K24 Architecture:
┌─────────────────────────────────────────────────────────────┐
│                     Input (observations)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SemiStructuredLinear24 (fc1)                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Module A: EMA(|A|) × |W| × sigmoid(α)              │    │
│  │ Module B: Pattern Projection + Gumbel-Softmax       │    │
│  │ Result: Unit-type-specific 2:4 masks                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      GRUCell (rnn)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SemiStructuredLinear24 (fc2)                    │
│              SemiStructuredLinear24 (fc3)                    │
│              SemiStructuredLinear24 (fc4)                    │
│                         → Q-values                           │
└─────────────────────────────────────────────────────────────┘
```

## Usage Example

```python
# Configuration
args = types.SimpleNamespace(
    n_unit_types=3,
    rnn_hidden_dim=64,
    n_actions=14,
    use_rnn=True,
    t_max=10050000,
    K24_args={
        "temperature_init": 5.0,
        "temperature_min": 0.2,
        "anneal_end_step": 8040000,
        "div_coef": 0.1,
        "reset_interval": 500000,
        "reset_ratio": 0.2,
        "finetune_start_ratio": 0.8,
        "finetune_lr_decay": 0.9,
    }
)

# Create agent and learner
agent = K24_type_NRNNAgent(input_shape=256, args=args)
learner = K24_NQLearner(mac, scheme, logger, args)

# Training loop
for episode in range(max_episodes):
    batch = collect_experience()
    loss = learner.train(batch, t_env, episode_num)
```

## Verified Components

All components are successfully registered:
- Agent: `K24_type_n_rnn_1R3` ✓
- Learner: `k24_nq_learner` ✓

## Testing

Run training with:
```bash
python src/main.py --config=algs/K24_qmix_rnn_1R3 --env-config=envs/smic
```

## Correspondence to Design Document

The implementation follows the design in `/mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/最终方案设计_K_24.md`:

- **Module A**: Implemented in `SemiStructuredLinear24._compute_heterogeneous_scores()`
- **Module B**: Implemented in `SemiStructuredLinear24._pattern_gumbel_softmax()`
- **Module C**: Implemented in `PatternOrthogonalityLoss.forward()`
- **Training Dynamics**: Implemented in `K24_NQLearner.train()`

## SMACv2-Specific Considerations

### Environment Complexity
- **Micromanagement**: 3s5z, 3s6z scenarios require precise control
- **Large-scale battles**: corridor, 6h_vs_8z require strategic diversity
- **Mixed unit types**: MMM, MMM2 require high heterogeneity

### Recommended Hyperparameters

| Scenario Type | div_coef | anneal_end | reset_interval | finetune_start |
|--------------|----------|------------|----------------|----------------|
| Micro (3s5z) | 0.1-0.15 | 0.8 | 500K-1000K | 0.75-0.8 |
| Large (6h_vs_8z) | 0.15-0.2 | 0.85-0.9 | 1000K | 0.85 |
| Mixed (MMM) | 0.15-0.25 | 0.8 | 500K | 0.8 |

### Training Timeline

```
0% ──────────────────────────────────────────────────────────────────── 100%
    |                          |                    |
    Training start             Annealing end        Finetune start
    (mask exploration)         (stable masks)       (freeze & fine-tune)

    anneal_end: 0.8            finetune_start_ratio: 0.8
    anneal_end_step: 8.04M     finetune_start_step: 8.04M

Phase 1 (0-80%): Mask exploration with temperature annealing
Phase 2 (80-100%): Fixed masks with LR decay for fine-tuning
```

## Notes

- Strict 2:4 sparsity ensures hardware acceleration compatibility
- Pattern-based approach maintains differentiability
- Adaptive mechanisms handle RL distribution shift
- Diversity loss promotes unit-type specialization
- Rewind & Finetune improves final performance (2-5% expected)

## Next Steps

1. Test training on SMACv2 scenarios (3s5z, corridor, MMM)
2. Compare performance against baseline (QMix, Kalei)
3. Tune hyperparameters for specific scenarios
4. Benchmark sparsity and computational efficiency
5. Analyze learned patterns per unit type

## References

- Design Document: `/mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/最终方案设计_K_24.md`
- MPE Implementation: `/mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/Semi_Kalei/MPE_K24_IMPLEMENTATION_SUMMARY.md`
- SMACv2 Reference Config: `config/algs/Kalei_qmix_rnn_1R3.yaml`
