"""Load weights directly without Transformers' optional bulk GPU warmup."""
from threading import RLock

_LOAD_LOCK = RLock()


def load_pretrained(model_class, *args, **kwargs):
    """Avoid one model-sized scratch allocation; retain normal weight loading.

    Transformers 4.55.4's allocator warmup can OOM before loading weights that
    fit normally. It is only an allocation optimization, not model execution.
    Serialize the temporary override and restore it even if loading fails.
    Actual weight allocation errors are deliberately not caught or retried.
    """
    from transformers import modeling_utils
    with _LOAD_LOCK:
        original = modeling_utils.caching_allocator_warmup
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        try:
            return model_class.from_pretrained(*args, **kwargs)
        finally:
            modeling_utils.caching_allocator_warmup = original
