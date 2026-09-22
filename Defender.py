"""Compatibility imports for defense policies.

Implementation lives in dlm_steering; this module preserves existing imports.
"""
from dlm_steering.defenses import (
    DEFENDERS,
    Ours,
    V3,
    SelfReminder,
    DiffuGuard,
)
from dlm_steering.defenses.base import (
    _prompt_text_mask,
    _hidden,
    _replace,
    _GateReached,
    _GatePass,
    _ForwardPlan,
    _StepScalars,
    _PendingAudit,
    _BoundaryReading,
    Defender,
    NullDefender,
)
