# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Patch for https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/instantiate/_instantiate2.py
# until https://github.com/facebookresearch/hydra/issues/2140 is resolved

import importlib
import inspect
from enum import Enum
from typing import Any

from omegaconf import OmegaConf

try:
    from megatron.training.config.instantiate_utils import (
        InstantiationException,
        InstantiationMode,  # noqa: F401  (re-exported for tests / external callers)
        _call_target,  # noqa: F401  (re-exported for tests / external callers)
        _convert_node,  # noqa: F401  (re-exported for tests / external callers)
        _convert_target_to_string,  # noqa: F401  (re-exported for tests / external callers)
        _extract_pos_args,  # noqa: F401  (re-exported for tests / external callers)
        _filter_kwargs_for_target,  # noqa: F401  (re-exported for tests / external callers)
        _is_target,  # noqa: F401  (re-exported for tests / external callers)
        _Keys,  # noqa: F401  (re-exported for tests / external callers)
        _locate,  # noqa: F401  (re-exported for tests / external callers)
        _prepare_input_dict_or_list,  # noqa: F401  (re-exported for tests / external callers)
        _resolve_target,  # noqa: F401  (re-exported for tests / external callers)
        instantiate,  # noqa: F401  (re-exported for tests / external callers)
        instantiate_node,  # noqa: F401  (re-exported for tests / external callers)
        target_allowlist,
    )
except ImportError:

    class InstantiationException(Exception):
        """Exception raised when config instantiation fails."""


    class InstantiationMode(Enum):
        """Fallback instantiation mode for conversion-only environments."""

        STRICT = "strict"
        LENIENT = "lenient"


    class _Keys:
        TARGET = "_target_"
        ARGS = "_args_"
        RECURSIVE = "_recursive_"
        CONVERT = "_convert_"
        PARTIAL = "_partial_"


    class _TargetAllowlist:
        def __init__(self) -> None:
            self.allowed_prefixes: set[str] = set()

        def add_prefix(self, prefix: str) -> None:
            self.allowed_prefixes.add(prefix)

        def is_allowed(self, target: str) -> bool:
            return any(target.startswith(prefix) for prefix in self.allowed_prefixes)


    target_allowlist = _TargetAllowlist()


    def _convert_target_to_string(target: Any) -> str:
        if isinstance(target, str):
            return target
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if module and qualname:
            return f"{module}.{qualname}"
        raise InstantiationException(f"Cannot convert target to import string: {target!r}")


    def _locate(path: str) -> Any:
        module_name, _, attr = path.rpartition(".")
        if not module_name:
            raise InstantiationException(f"Invalid target path: {path!r}")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except (ImportError, AttributeError) as exc:
            raise InstantiationException(f"Could not locate target {path!r}") from exc


    def _resolve_target(target: Any, full_key: str = "") -> Any:
        target = _convert_target_to_string(target)
        _validate_target_prefix(target=target, full_key=full_key)
        return _locate(target)


    def _is_target(value: Any) -> bool:
        return isinstance(value, dict) and _Keys.TARGET in value


    def _prepare_input_dict_or_list(config: Any) -> Any:
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
        return config


    def _convert_node(node: Any, *args: Any, **kwargs: Any) -> Any:
        return _prepare_input_dict_or_list(node)


    def _extract_pos_args(config: dict[str, Any]) -> list[Any]:
        args = config.get(_Keys.ARGS, [])
        if not isinstance(args, list):
            raise InstantiationException(f"{_Keys.ARGS} must be a list, got {type(args).__name__}")
        return args


    def _filter_kwargs_for_target(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return kwargs
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs
        return {name: value for name, value in kwargs.items() if name in signature.parameters}


    def _call_target(target: Any, *args: Any, **kwargs: Any) -> Any:
        return target(*args, **kwargs)


    def instantiate_node(config: Any, *args: Any, mode: InstantiationMode = InstantiationMode.LENIENT, **kwargs: Any) -> Any:
        return instantiate(config, *args, mode=mode, **kwargs)


    def instantiate(config: Any, *args: Any, mode: InstantiationMode = InstantiationMode.LENIENT, **kwargs: Any) -> Any:
        config = _prepare_input_dict_or_list(config)
        if isinstance(config, list):
            return [instantiate(item, mode=mode) for item in config]
        if not isinstance(config, dict):
            return config
        if _Keys.TARGET not in config:
            return {key: instantiate(value, mode=mode) for key, value in config.items()}

        target = _resolve_target(config[_Keys.TARGET])
        pos_args = [instantiate(arg, mode=mode) for arg in _extract_pos_args(config)]
        pos_args.extend(args)
        call_kwargs = {
            key: instantiate(value, mode=mode)
            for key, value in config.items()
            if not key.startswith("_")
        }
        call_kwargs.update(kwargs)
        if mode == InstantiationMode.LENIENT:
            call_kwargs = _filter_kwargs_for_target(target, call_kwargs)
        return _call_target(target, *pos_args, **call_kwargs)


_ALLOWED_TARGET_PREFIXES: set[str] = {
    "megatron.",
    "torch.",
    "nvidia.",
    "transformers.",
    "numpy.",
    "nemo.",
}


# Mirror Bridge's allowlist into the MLM `target_allowlist` singleton, which is
# the source of truth consulted by `_validate_target_prefix` below. MLM's
# default prefixes are narrower (megatron.training./megatron.core./torch./
# transformers./signal.) and would otherwise reject e.g. `megatron.bridge.*`,
# `nvidia.*`, `numpy.*`, `nemo.*`.
def _as_module_prefix(prefix: str) -> str:
    """Ensure prefix ends with '.' so allowlist matches at module boundaries."""
    return prefix if prefix.endswith(".") else prefix + "."


def _seed_allowlist() -> None:
    for prefix in _ALLOWED_TARGET_PREFIXES:
        target_allowlist.add_prefix(_as_module_prefix(prefix))


_seed_allowlist()


def register_allowed_target_prefix(prefix: str) -> None:
    """Register an additional allowed module prefix for _target_ instantiation.

    This allows extending the default allowlist for use cases that require
    instantiating classes from other packages.
    """
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"Prefix must be a non-empty string, got {prefix!r}")
    _ALLOWED_TARGET_PREFIXES.add(prefix)
    # MLM's `target_allowlist` is the source of truth for `_validate_target_prefix`
    # and requires the trailing dot to match at module boundaries.
    target_allowlist.add_prefix(_as_module_prefix(prefix))


def _validate_target_prefix(*, target: str, full_key: str) -> None:
    """Validate that a _target_ string is permitted by the allowlist."""
    if not target_allowlist.is_allowed(target):
        raise InstantiationException(
            f"Instantiation of '{target}' is not allowed. "
            f"The target must start with one of the allowed prefixes: "
            f"{sorted(target_allowlist.allowed_prefixes)}. "
            f"Use register_allowed_target_prefix() to add additional allowed prefixes."
            + (f"\nfull_key: {full_key}" if full_key else "")
        )
