_MODEL_REGISTRY = {}

def register_model(name):
    """
    Decorator to register a model class.
    Usage:
        @register_model("model_name")
        class MyModel(nn.Module): ...
    """
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model_class(name):
    """Retrieves a model class from the registry."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]
