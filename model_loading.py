from threading import RLock

_LOAD_LOCK = RLock()


def load_pretrained(model_class, *args, **kwargs):
    from transformers import modeling_utils
    with _LOAD_LOCK:
        original = modeling_utils.caching_allocator_warmup
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        try:
            return model_class.from_pretrained(*args, **kwargs)
        finally:
            modeling_utils.caching_allocator_warmup = original
