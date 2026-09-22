"""Compatibility imports for attack policies.

Implementation lives in dlm_steering; this module preserves existing imports.
"""
from dlm_steering.attacks import (
    ATTACKERS,
    DIJA,
    PAP,
    PAIR,
)
from dlm_steering.attacks.base import (
    AttackResult,
    Attacker,
    NoAttack,
    Prefix,
    _assistant_text,
    _default_attack_device,
)
from dlm_steering.attacks.dija import (
    DIJA_MASK_PATTERN,
    DIJA_REFINED,
)
