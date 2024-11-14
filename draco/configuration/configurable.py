import functools
import inspect
from typing import Any, Callable

from omegaconf import DictConfig

__all__ = ["configurable"]


def _called_with_cfg(*args, **kwargs) -> bool:
    """
    Check if the function is called with a `DictConfig` as the first argument.

    Returns:
        (bool): whether the function is called with a `DictConfig` as the first argument.
            Or the `cfg` keyword argument is a `DictConfig`.
    """

    if len(args) > 0 and isinstance(args[0], DictConfig):
        return True
    if isinstance(kwargs.get("cfg", None), DictConfig):
        return True
    return False


def _get_args_from_cfg(from_config_func: Callable[[Any], dict[str, Any]], *args, **kwargs) -> dict[str, Any]:
    """
    Get the input arguments of the decorated function from a `DictConfig` object.

    Returns:
        (dict): The input arguments of the class `__init__` method.
    """

    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        raise ValueError("The first argument of `{}` must be named as `cfg`.".format(from_config_func.__name__))

    # Forwarding all arguments to `from_config`, if the arguments of `from_config` are only `*args` or `*kwargs`.
    if any(param.kind in [param.VAR_POSITIONAL or param.VAR_KEYWORD] for param in signature.parameters.values()):
        result = from_config_func(*args, **kwargs)

    # If there is any positional arguments.
    else:
        positional_args_name = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in kwargs.keys():
            if name not in positional_args_name:
                extra_kwargs[name] = kwargs.pop(name)
        result = from_config_func(*args, **kwargs)
        # These args are forwarded directly to `__init__` method.
        result.update(extra_kwargs)

    return result


def configurable(init_func: Callable = None, *, from_config: Callable[[Any], dict[str, Any]] | None = None) -> Callable:
    """
    A decorator of a function or a class `__init__` method,
    to make it configurable by a `DictConfig` object.

    Example:
    ```python
    # 1. Decorate a function.
    @configurable(from_config=lambda cfg: { "x": cfg.x })
    def func(x, y=2, z=3):
        pass

    a1 = func(x=1, y=2) # Call with regular args.
    a2 = func(cfg) # Call with a `DictConfig` object.
    a3 = func(cfg, y=2, z=3) # Call with a `DictConfig` object and regular arguments.

    # 2. Decorate a class `__init__` method.
    class A:
        @configurable
        def __init__(self, *args, **kwargs) -> None:
            pass

        @classmethod
        def from_config(cls, cfg) -> dict:
            pass

    a1 = A(x, y) # Call with regular constructor.
    a2 = A(cfg) # Call with a `DictConfig` object.
    a3 = A(cfg, x, y) # Call with a `DictConfig` object and regular arguments.
    ```

    Args:
        `init_func` (callable): a function or a class method.
        `from_config` (callable): a function that converts a `DictConfig` to the
            input arguments of the decorated function.
            It is always required.
    """

    # Decorating a function
    if init_func is None:
        # Prevent common misuse: `@configurable()`.
        if from_config is None:
            return configurable

        assert inspect.isfunction(from_config), "`from_config` must be a function."

        def wrapper(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_cfg(from_config, *args, **kwargs)
                    return func(**explicit_args)
                else:
                    return func(*args, **kwargs)

            wrapped.from_config = from_config
            return wrapped

        return wrapper

    # Decorating a class `__init__` method
    else:
        assert(
            inspect.isfunction(init_func) and from_config is None and init_func.__name__ == "__init__"
        ), "Invalid usage of @configurable."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = getattr(self, "from_config")
            except AttributeError as e:
                raise AttributeError("Class with `@configurable` should have a `from_config` classmethod.") from e

            if not inspect.ismethod(from_config_func):
                raise AttributeError("Class with `@configurable` should have a `from_config` classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_cfg(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped
