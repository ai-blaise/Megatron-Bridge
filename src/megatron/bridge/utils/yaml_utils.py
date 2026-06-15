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

import enum
import functools
from contextlib import contextmanager
from typing import Any, Optional

import yaml

try:
    from megatron.training.config.yaml_utils import (
        _enum_representer,  # noqa: F401
        _function_representer,  # noqa: F401
        _generation_config_representer,  # noqa: F401
        _partial_representer,  # noqa: F401
        _pretrained_config_representer,  # noqa: F401
        _safe_object_representer,  # noqa: F401
        _torch_dtype_representer,  # noqa: F401
        safe_yaml_representers,
    )
except ImportError:

    def _safe_object_representer(dumper: yaml.SafeDumper, data: Any):
        if hasattr(data, "to_dict"):
            return dumper.represent_dict(data.to_dict())
        if hasattr(data, "__dict__"):
            return dumper.represent_dict(data.__dict__)
        return dumper.represent_str(str(data))


    def _enum_representer(dumper: yaml.SafeDumper, data: enum.Enum):
        return dumper.represent_data(data.value)


    def _function_representer(dumper: yaml.SafeDumper, data: Any):
        module = getattr(data, "__module__", "")
        qualname = getattr(data, "__qualname__", getattr(data, "__name__", str(data)))
        return dumper.represent_str(f"{module}.{qualname}" if module else qualname)


    def _partial_representer(dumper: yaml.SafeDumper, data: functools.partial):
        return dumper.represent_str(repr(data))


    def _torch_dtype_representer(dumper: yaml.SafeDumper, data: Any):
        return dumper.represent_str(str(data))


    _pretrained_config_representer = _safe_object_representer
    _generation_config_representer = _safe_object_representer


    @contextmanager
    def safe_yaml_representers():
        """Temporarily register Bridge-safe YAML representers."""
        representers = {
            enum.Enum: _enum_representer,
            functools.partial: _partial_representer,
        }
        previous = dict(yaml.SafeDumper.yaml_representers)
        previous_multi = dict(yaml.SafeDumper.yaml_multi_representers)
        try:
            for typ, representer in representers.items():
                yaml.SafeDumper.add_multi_representer(typ, representer)
            yaml.SafeDumper.add_multi_representer(type(lambda: None), _function_representer)
            yaml.SafeDumper.add_multi_representer(object, _safe_object_representer)
            yield
        finally:
            yaml.SafeDumper.yaml_representers = previous
            yaml.SafeDumper.yaml_multi_representers = previous_multi


def dump_dataclass_to_yaml(obj: Any, filename: Optional[str] = None) -> Optional[str]:
    """Dump a dataclass object or other Python object to a YAML file or string.

    Uses safe representers to handle common types.

    Args:
        obj: The object to dump.
        filename: If provided, the path to the file where YAML should be written.
                  If None, returns the YAML string directly.

    Returns:
        If filename is None, returns the YAML string representation of the object.
        Otherwise, returns None.
    """
    with safe_yaml_representers():
        if filename is not None:
            with open(filename, "w+") as f:
                yaml.safe_dump(obj, f)
        else:
            return yaml.safe_dump(obj)
