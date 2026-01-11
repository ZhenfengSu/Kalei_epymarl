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

