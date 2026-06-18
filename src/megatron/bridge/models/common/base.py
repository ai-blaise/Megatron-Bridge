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

from collections.abc import Callable
from typing import Any, TypeVar

try:
    from megatron.training.models.base import (
        BuildConfigT,  # noqa: F401
        ModelBuilder,  # noqa: F401
        ModelConfig,  # noqa: F401
        ModelT,  # noqa: F401
        Serializable,  # noqa: F401
        compose_hooks,  # noqa: F401
    )
except ImportError:
    BuildConfigT = TypeVar("BuildConfigT")
    ModelT = TypeVar("ModelT")

    class Serializable:
        """Fallback serialization mixin used when Megatron training is unavailable."""

        def to_dict(self) -> dict[str, Any]:
            return dict(self.__dict__)

        @classmethod
        def from_dict(cls, state: dict[str, Any]) -> "Serializable":
            obj = cls.__new__(cls)
            obj.__dict__.update(state)
            return obj

    class ModelConfig(Serializable):
        """Fallback model config base class used for conversion-only imports."""

        def finalize(self) -> None:
            return None

    class ModelBuilder(Serializable):
        """Fallback builder base class used for conversion-only imports."""

        def __init__(self, model_config: ModelConfig) -> None:
            self._model_config = model_config

        @classmethod
        def __class_getitem__(cls, _params):
            """Accept generic subscripting in newer Bridge builder annotations."""
            return cls

    def compose_hooks(hooks: list[Callable[..., Any]] | None):
        """Compose a hook list into a single callable when training helpers are absent."""
        if not hooks:
            return None

        def composed(value):
            for hook in hooks:
                value = hook(value)
            return value

        return composed
