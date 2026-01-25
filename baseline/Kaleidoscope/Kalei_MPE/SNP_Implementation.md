# SNP Algorithm Implementation for MPE

## Overview

This document describes the implementation of the **SNP (Structured Network Pruning for Parameter Sharing)** algorithm for the MPE (Multi-Agent Particle Environment) project, based on the Kaleidoscope codebase. SNP-PS is a parameter sharing method for multi-agent reinforcement learning that uses structured network pruning to create diverse subnetworks for different agents within a shared root network.

## Theoretical Background

### Lottery Group Ticket Hypothesis (LGTH)

SNP is inspired by the Lottery Ticket Hypothesis, which states that dense, randomly initialized neural networks contain subnetworks (winning tickets) that can match the performance of the original network when trained in isolation. SNP extends this to multi-agent systems with the **Lottery Group Ticket Hypothesis**:

> In a sufficiently large dense network, there exist subnetworks that can provide both identifiability (agent-specific behaviors) and performance comparable to full parameter sharing.

### Core Mechanism

SNP uses **structured network pruning** to create binary masks for each agent:

1. **Initialization**: A single dense neural network (root network) is initialized
2. **Mask Generation**: Random binary masks are generated for each agent using structured pruning
3. **Subnetwork Construction**: Each agent uses `f(x; theta element-wise_multiply M_i)` where `M_i` is agent i's mask
4. **Parameter Sharing**:
   - **Overlapping regions**: Multiple agents share the same parameters (high sample efficiency)
   - **Non-overlapping regions**: Agent-specific parameters (high representational capacity)

### Advantages Over Other Methods

Compared to baseline methods:

- **vs. Full Parameter Sharing**: SNP provides agent identifiability, allowing agents to learn different behaviors
- **vs. One-Hot + Parameter Sharing**: SNP doesn't require additional input dimensions
- **vs. SePS (Selective Parameter Sharing)**: SNP doesn't need extra clustering networks or parameters
- **Controllable**: The pruning ratio directly controls the degree of parameter sharing

## Implementation Details

### Files Modified/Created

1. **`src/modules/agents/snp_rnn_agent.py`** (NEW)
   - `SNP_RNNAgent_1R3`: Main SNP agent implementation for MPE
   - Structured pruning with binary masks for each layer
   - Mask generation using random sampling with specified sparsity ratios

2. **`src/modules/agents/__init__.py`** (MODIFIED)
   - Registered `SNP_RNNAgent_1R3` as `"SNP_rnn_1R3"`

3. **`src/controllers/SNP_controller.py`** (NEW)
   - `SNP_MAC`: Multi-agent controller for SNP in MPE
   - Handles agent identification and mask selection

4. **`src/controllers/__init__.py`** (MODIFIED)
   - Registered `SNP_MAC` as `"SNP_mac"`

5. **`src/config/algs/SNP_qmix_rnn_1R3.yaml`** (NEW)
   - Configuration file for SNP-QMIX algorithm
   - Sparsity ratio settings for different layers

### Key Differences from SMACv2 Implementation

| Aspect | SMACv2 Implementation | MPE Implementation |
|--------|----------------------|--------------------|
| **Agent Count** | `n_unit_types` | `n_agents` |
| **Agent IDs** | Extracted from observation (unit types) | Direct (0 to n_agents-1) |
| **Controller** | `SNP_type_NMAC` (extends `NMAC`) | `SNP_MAC` (extends `BasicMAC`) |
| **ID Extraction** | One-hot encoding from obs | `torch.arange` for simple IDs |
| **Environment** | SMAC (StarCraft II) | MPE (Particle World) |

### Agent Architecture

The SNP agent uses a 4-layer architecture:

```
Input -> fc1 -> RNN (GRU) -> fc2 -> fc3 -> fc4 -> Q-values
         (no mask)         (mask_0) (mask_1) (mask_2)
```

- **fc1**: Standard linear layer (no pruning)
- **RNN**: GRU cell for temporal dependencies (no pruning)
- **fc2, fc3, fc4**: Linear layers with **structured pruning masks**

### Mask Application

Masks are applied element-wise to neuron activations:

```python
# Example: mask applied to fc2 layer
h = h * self.mask_0[agent_ids]  # Zero out pruned neurons
q = self.fc2(h)                # Linear transformation
q = F.relu(q)                  # Activation
```

This is different from Kaleidoscope's approach:
- **Kaleidoscope**: Learns sparse thresholds via gradient descent
- **SNP**: Uses fixed random masks generated at initialization

### Key Methods in SNP Agent

#### 1. `__init__(input_shape, args)`
- Initializes network layers
- Generates binary masks using `th.rand() > sparsity_ratio`
- Registers masks as buffers (not parameters) to save in state_dict

#### 2. `forward(inputs, hidden_state, agent_ids)`
- Standard forward pass with mask application
- `agent_ids` determines which mask each agent uses
- Masks are applied before activation functions

#### 3. `get_sparsities()`
- Calculates actual sparsity ratios for each layer
- Returns overlap statistics (how many agents share each neuron)

#### 4. `get_mask_diversity()`
- Measures pairwise differences between agent masks
- Useful for analyzing mask diversity and overlap

## Configuration

### SNP Arguments

The SNP configuration in `SNP_qmix_rnn_1R3.yaml` includes:

```yaml
SNP_args:
  layers_sparsities:
    - 0.5  # fc2: 50% neurons pruned
    - 0.5  # fc3: 50% neurons pruned
    - 0.5  # fc4: 50% neurons pruned
```

### Sparsity Ratio Guidelines

- **0.3 - 0.4 (Low)**: More parameter sharing, better sample efficiency
  - Use when agents have similar roles
  - Faster convergence
  - Less agent specialization

- **0.5 - 0.6 (Medium)**: Balanced configuration
  - Good starting point for most scenarios
  - Mix of shared and specialized parameters

- **0.7 - 0.8 (High)**: Less sharing, more agent-specific capacity
  - Use when agents have highly distinct roles
  - More diverse behaviors
  - May require more training samples

## Usage

### Basic Training Command

```bash
cd /mnt/lc_gpu_test/Semi_Kaleidoscope/Kalei_epymarl/baseline/Kaleidoscope/Kalei_MPE

python src/run.py --config=SNP_qmix_rnn_1R3 --env-config=gymma env_args.env_name=MPE.envs.simple_spread_v3
```

### Running on Different MPE Environments

```bash
# Example: Run on simple_push scenario
python src/run.py --config=SNP_qmix_rnn_1R3 --env-config=gymma env_args.env_name=MPE.envs.simple_push_v3

# Example: Run on simple_adversary scenario
python src/run.py --config=SNP_qmix_rnn_1R3 --env-config=gymma env_args.env_name=MPE.envs.simple_adversary_v3

# Example: Run on simple_tag scenario
python src/run.py --config=SNP_qmix_rnn_1R3 --env-config=gymma env_args.env_name=MPE.envs.simple_tag_v3
```

### Custom Sparsity Configuration

Create a custom config file or modify existing:

```yaml
SNP_args:
  layers_sparsities:
    - 0.4  # Less pruning in fc2
    - 0.6  # More pruning in fc3
    - 0.5  # Medium pruning in fc4
```

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **Sparsity ratios** (`layers_sparsities`): Main SNP parameter
   - Start with `[0.5, 0.5, 0.5]`
   - Adjust based on task complexity

2. **Learning rate** (`lr`): Standard QMIX parameter
   - Default: 0.0005 (usually set in default.yaml)
   - Reduce if training is unstable

3. **Epsilon annealing** (`epsilon_anneal_time`):
   - Default: 50000
   - Increase for more exploration

## Implementation Details

### SNP_MAC Controller

The `SNP_MAC` controller extends `BasicMAC` with these modifications:

1. **`_build_inputs()`** method:
   - Builds inputs similarly to `BasicMAC`
   - Generates agent IDs using `torch.arange(n_agents)`
   - Returns both inputs and agent IDs

2. **`forward()`** method:
   - Passes agent IDs to the SNP agent
   - SNP agent uses agent IDs to select appropriate masks

### Comparison with Kalei_MAC

| Aspect | Kalei_MAC | SNP_MAC |
|--------|-----------|---------|
| **Agent ID Handling** | Uses obs_agent_id one-hot | Uses direct agent IDs |
| **Mask Learning** | Learnable thresholds | Fixed random masks |
| **Base Class** | BasicMAC | BasicMAC |
| **Properties** | `sparsities`, `mask_parameters` | None (no learnable masks) |

## Code Reference

### SNP Agent Implementation

```python
class SNP_RNNAgent_1R3(nn.Module):
    def __init__(self, input_shape, args):
        # ... initialize layers ...

        # Generate masks for 3 layers (fc2, fc3, fc4)
        for i, layer_sparsity in enumerate(self.sparsity_ratios):
            self.register_buffer(
                f"mask_{i}",
                th.rand(self.n_agents, hidden_dim) > layer_sparsity
            )

    def forward(self, inputs, hidden_state, agent_ids):
        # ... standard layers ...
        h = h * self.mask_0[agent_ids]
        q = self.fc2(h) * self.mask_1[agent_ids]
        # ... etc
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `snp_rnn_agent.py` is in the correct directory
   ```bash
   ls src/modules/agents/snp_rnn_agent.py
   ```

2. **Config Not Found**: Verify the config file path
   ```bash
   ls src/config/algs/SNP_qmix_rnn_1R3.yaml
   ```

3. **Agent Dimension Mismatch**: Ensure `n_agents` matches your environment
   - MPE environments typically use 2-5 agents
   - Check your environment configuration

### Performance Tips

1. **Start with medium sparsity** (0.5) for most tasks
2. **Increase sparsity** if agents need more distinct behaviors
3. **Decrease sparsity** if sample efficiency is critical
4. **Monitor mask diversity** using `get_mask_diversity()` method

## Implementation Differences from Other Methods

| Aspect | Full PS | Kaleidoscope | SNP |
|--------|---------|--------------|-----|
| **Mask Generation** | N/A | Learned thresholds | Fixed random masks |
| **Mask Update** | N/A | Reset periodically | Static (never changed) |
| **Training Objective** | Standard RL | Diversity loss + RL | Standard RL only |
| **Sparsity Control** | N/A | Threshold parameters | Direct sparsity ratio |
| **Additional Parameters** | 0 | Threshold parameters | 0 |
| **Complexity** | Low | Higher | Low |

## References

- **SNP Paper**: Structured Network Pruning for Parameter Sharing in Multi-Agent Reinforcement Learning
- **Kaleidoscope Paper**: Sparse Neural Networks for Multi-Agent Reinforcement Learning
- **SMACv2 Implementation**: `Kalei_SMACv2/SNP_Implementation.md`

## Summary

The SNP implementation for MPE provides:
- Simpler agent identification (direct IDs vs unit types)
- Fixed random masks for parameter sharing
- Controllable sparsity via configuration
- No additional learnable parameters
- Compatible with MPE environments (simple_spread, simple_push, simple_tag, etc.)
- Easy integration with existing QMIX framework

The implementation follows the specifications from the SNP paper and adapts the SMACv2 implementation for the MPE environment structure.
