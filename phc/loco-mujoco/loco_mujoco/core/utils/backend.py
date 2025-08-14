from types import ModuleType

import numpy as np
import jax.numpy as jnp


def assert_backend_is_supported(module: ModuleType):
    """
    Check if the given module is supported.

    Args:
        module (ModuleType): The module to check (e.g., numpy or jax.numpy).

    Returns:
        bool: True if the module is supported, False otherwise.
    """
    is_supporter = module in {np, jnp}
    assert is_supporter, f"Unsupported backend module: {module.__name__}"

