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

import logging
from dataclasses import fields, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

try:
    from megatron.training.config.container import ConfigContainerBase as _ConfigContainerBase  # noqa: F401
    from megatron.training.config.utils import (
        _get_init_false_fields,  # noqa: F401
        _resolve_target_class,  # noqa: F401
    )
    from megatron.training.config.utils import (
        sanitize_dataclass_config as _sanitize_dataclass_config,
    )
except (ImportError, ModuleNotFoundError):

    class _ConfigContainerBase:
        """Small fallback for Megatron-LM versions without training.config.container."""

        @staticmethod
        def _convert_value_to_dict(value: Any) -> Any:
            if is_dataclass(value):
                return {
                    field.name: _ConfigContainerBase._convert_value_to_dict(getattr(value, field.name))
                    for field in fields(value)
                }
            if hasattr(value, "to_dict"):
                return value.to_dict()
            if isinstance(value, list):
                return [_ConfigContainerBase._convert_value_to_dict(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_ConfigContainerBase._convert_value_to_dict(item) for item in value)
            if isinstance(value, dict):
                return {key: _ConfigContainerBase._convert_value_to_dict(item) for key, item in value.items()}
            return value

        @staticmethod
        def _convert_value_to_yaml_safe(value: Any) -> Any:
            value = _ConfigContainerBase._convert_value_to_dict(value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, list):
                return [_ConfigContainerBase._convert_value_to_yaml_safe(item) for item in value]
            if isinstance(value, tuple):
                return [_ConfigContainerBase._convert_value_to_yaml_safe(item) for item in value]
            if isinstance(value, dict):
                return {
                    str(key): _ConfigContainerBase._convert_value_to_yaml_safe(item) for key, item in value.items()
                }
            return str(value)

        def to_dict(self) -> dict[str, Any]:
            return self._convert_value_to_dict(self)

        def to_yaml(self, path: str | Path) -> None:
            import yaml

            with open(path, "w", encoding="utf-8") as stream:
                yaml.safe_dump(self._convert_value_to_yaml_safe(self), stream, sort_keys=False)

    def _resolve_target_class(config_dict: Mapping[str, Any]) -> type[Any] | None:
        target = config_dict.get("_target_")
        if not isinstance(target, str) or "." not in target:
            return None

        module_name, class_name = target.rsplit(".", 1)
        try:
            module = import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            return None
        return getattr(module, class_name, None)

    def _get_init_false_fields(target_class: type[Any] | None) -> set[str]:
        if target_class is None or not is_dataclass(target_class):
            return set()
        return {field.name for field in fields(target_class) if not field.init}

    def _sanitize_dataclass_config(config_dict: dict[str, Any]) -> dict[str, Any]:
        target_class = _resolve_target_class(config_dict)
        init_false_fields = _get_init_false_fields(target_class)

        sanitized: dict[str, Any] = {}
        for key, value in config_dict.items():
            if key in init_false_fields:
                continue
            if isinstance(value, dict):
                sanitized[key] = _sanitize_dataclass_config(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    _sanitize_dataclass_config(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized


logger = logging.getLogger(__name__)


def create_ddp_config(
    wrap_with_ddp: bool = True,
    use_distributed_optimizer: bool = True,
    use_megatron_fsdp: bool = False,
    overrides: Mapping[str, object] | None = None,
    finalize: bool = True,
) -> object | None:
    """Create a finalized Bridge DDP config for external model construction."""
    if not wrap_with_ddp:
        return None

    from megatron.bridge.training.config import DistributedDataParallelConfig

    ddp_config = {
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if use_megatron_fsdp:
        ddp_config.update(
            {
                "use_distributed_optimizer": True,
                "check_for_nan_in_grad": True,
                "use_megatron_fsdp": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "overlap_grad_reduce": True,
            }
        )
    ddp_config.update(overrides or {})

    config = DistributedDataParallelConfig(**ddp_config)
    if finalize:
        config.finalize()
    return config


def apply_run_config_backward_compat(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply backward compatibility transformations to run config.

    This function handles dataclass config fields that should not be passed to
    the constructor when loading older checkpoints. It automatically detects
    init=False fields by inspecting the target class.

    The entire config is sanitized recursively to handle init=False fields in any part of the configuration hierarchy.

    Args:
        config_dict: The full run configuration dictionary.

    Returns:
        The config dictionary with backward compatibility fixes applied.
    """
    return _sanitize_dataclass_config(config_dict)
