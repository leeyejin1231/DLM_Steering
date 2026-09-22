"""Compatibility imports for evaluation cache.

Implementation lives in dlm_steering; this module preserves existing imports.
"""
from dlm_steering.evaluation.cache import (
    sha256,
    saved_scope,
    matches,
    find_cached,
    load_cached,
    show_summary,
    show_cached,
)
