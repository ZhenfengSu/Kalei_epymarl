from .k24_pattern_gumbel_layer import (
    EMAActivationTracker,
    Pattern24Matrix,
    SemiStructuredLinear24,
    create_k24_linear,
)
from .k24_diversity import (
    PatternOrthogonalityLoss,
    K24DiversityManager,
    LayerPatternTracker,
)


REGISTRY = {}
REGISTRY["ema_activation_tracker"] = EMAActivationTracker
REGISTRY["pattern24_matrix"] = Pattern24Matrix
REGISTRY["semi_structured_linear24"] = SemiStructuredLinear24
REGISTRY["pattern_orthogonality_loss"] = PatternOrthogonalityLoss
REGISTRY["k24_diversity_manager"] = K24DiversityManager
REGISTRY["layer_pattern_tracker"] = LayerPatternTracker
