"""
Custom layer modules for LDSA with K24 2:4 sparsity.

This module implements semi-structured sparsity using the K-2:4 algorithm:
- Pattern-based 2:4 sparse masks
- Gumbel-Softmax for differentiable sampling
- Dynamic heterogeneous scoring with EMA activation tracking
- Pattern orthogonality diversity loss
"""

from .k24_sparse_linear import (
    EMAActivationTracker,
    Pattern24Matrix,
    SemiStructuredLinear24,
    create_k24_linear
)
from .k24_diversity import (
    PatternOrthogonalityLoss,
    K24DiversityManager,
    LayerPatternTracker
)

__all__ = [
    'EMAActivationTracker',
    'Pattern24Matrix',
    'SemiStructuredLinear24',
    'create_k24_linear',
    'PatternOrthogonalityLoss',
    'K24DiversityManager',
    'LayerPatternTracker'
]
