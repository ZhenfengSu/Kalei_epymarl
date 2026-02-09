"""
Attention modules for LDSA with TAAE-style integration.

This module implements attention mechanisms that integrate:
- Role embeddings (from subtask probabilities)
- Global state (environment-provided)
- Local observations

References:
- TAAE: Team-Aware Attention Extraction
- Multi-Head Attention for agent coordination
"""

from .role_state_attention import RoleStateAttention, GlobalStateFusion

__all__ = ['RoleStateAttention', 'GlobalStateFusion']
