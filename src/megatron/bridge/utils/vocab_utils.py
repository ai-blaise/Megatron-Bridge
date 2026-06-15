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

from functools import lru_cache

try:
    from megatron.training.vocab_utils import (
        _calculate_padded_vocab_size_cached,  # noqa: F401
        calculate_padded_vocab_size,  # noqa: F401
    )
except ImportError:

    @lru_cache(maxsize=None)
    def _calculate_padded_vocab_size_cached(
        vocab_size: int,
        make_vocab_size_divisible_by: int,
        tensor_model_parallel_size: int,
        add_extra_token_to_vocab: bool = False,
    ) -> int:
        """Calculate padded vocab size without importing Megatron training."""
        after = vocab_size + int(add_extra_token_to_vocab)
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while after % multiple != 0:
            after += 1
        return after


    def calculate_padded_vocab_size(
        vocab_size: int,
        make_vocab_size_divisible_by: int,
        tensor_model_parallel_size: int,
        add_extra_token_to_vocab: bool = False,
    ) -> int:
        """Pad vocab size for tensor-parallel divisibility."""
        return _calculate_padded_vocab_size_cached(
            vocab_size,
            make_vocab_size_divisible_by,
            tensor_model_parallel_size,
            add_extra_token_to_vocab,
        )
