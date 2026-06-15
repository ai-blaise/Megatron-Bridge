# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the GLM-4 bridge."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.glm.glm4_bridge import GLM4Bridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


pytestmark = pytest.mark.unit


@pytest.fixture
def dense_glm4_config():
    """Mock config for dense GLM-4."""
    return SimpleNamespace(
        architectures=["Glm4ForCausalLM"],
        attention_bias=True,
        attention_dropout=0.0,
        hidden_size=4096,
        hidden_act="silu",
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=131072,
        model_type="glm4",
        num_attention_heads=32,
        num_hidden_layers=40,
        num_key_value_heads=2,
        partial_rotary_factor=1.0,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        vocab_size=128815,
    )


@pytest.fixture
def omp_glm4_config():
    """Mock config for the OMP GLM-4 checkpoint."""
    return SimpleNamespace(
        architectures=["Glm4ForCausalLM"],
        attention_bias=True,
        attention_dropout=0.0,
        hidden_size=4096,
        hidden_act="silu",
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=131072,
        model_type="glm4",
        num_attention_heads=32,
        num_hidden_layers=40,
        num_key_value_heads=2,
        partial_rotary_factor=1.0,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        vocab_size=128815,
    )


@pytest.fixture
def dense_pretrained(dense_glm4_config):
    """Create a mock dense GLM-4 pretrained model."""
    model = Mock(spec=PreTrainedCausalLM)
    model.config = dense_glm4_config
    model.generation_config = Mock(spec=GenerationConfig)
    model.state = Mock()
    model.state.source = Mock()
    model.state.source.get_all_keys.return_value = []
    model.state.source.has_glob.side_effect = lambda pattern: pattern in {
        "*self_attn.q_norm.weight*",
        "*self_attn.k_norm.weight*",
        "*mlp.gate_proj.weight*",
        "*mlp.up_proj.weight*",
    }
    return model


@pytest.fixture
def omp_pretrained(omp_glm4_config):
    """Create a mock OMP GLM-4 pretrained model."""
    model = Mock(spec=PreTrainedCausalLM)
    model.config = omp_glm4_config
    model.generation_config = Mock(spec=GenerationConfig)
    model.state = Mock()
    model.state.source = Mock()
    model.state.source.get_all_keys.return_value = []
    model.state.source.has_glob.side_effect = lambda pattern: pattern in {
        "*mlp.gate_up_proj.weight*",
    }
    return model


class TestGLM4Bridge:
    """Test cases for GLM4Bridge."""

    def test_registration(self):
        """GLM4Bridge is registered as a MegatronModelBridge."""
        assert issubclass(GLM4Bridge, MegatronModelBridge)

    def test_provider_bridge_keeps_dense_qk_norms(self, dense_pretrained):
        """Dense GLM-4 keeps q/k layernorm enabled."""
        bridge = GLM4Bridge()
        provider = bridge.provider_bridge(dense_pretrained)

        assert isinstance(provider, GPTModelProvider)
        assert provider.hidden_size == dense_pretrained.config.hidden_size
        assert provider.num_layers == dense_pretrained.config.num_hidden_layers
        assert provider.qk_layernorm is True
        assert provider.params_dtype == torch.bfloat16

    def test_provider_bridge_disables_missing_qk_norms_for_omp(self, omp_pretrained):
        """OMP GLM-4 disables q/k layernorm when the HF checkpoint lacks those tensors."""
        bridge = GLM4Bridge()
        provider = bridge.provider_bridge(omp_pretrained)

        assert isinstance(provider, GPTModelProvider)
        assert provider.qk_layernorm is False

    def test_dense_mapping_registry_uses_separate_gate_and_up_proj(self, dense_pretrained):
        """Dense GLM-4 keeps the split gate/up MLP mapping."""
        bridge = GLM4Bridge()
        bridge.hf_pretrained = dense_pretrained
        bridge.hf_config = dense_pretrained.config

        registry = bridge.mapping_registry()
        mapping = registry.megatron_to_hf_lookup("decoder.layers.0.mlp.linear_fc1.weight")

        assert isinstance(mapping, GatedMLPMapping)
        assert mapping.hf_param["gate"] == "model.layers.0.mlp.gate_proj.weight"
        assert mapping.hf_param["up"] == "model.layers.0.mlp.up_proj.weight"

    def test_omp_mapping_registry_uses_fused_gate_up_proj(self, omp_pretrained):
        """OMP GLM-4 uses fused gate_up_proj weights and skips q/k layernorm tensors."""
        bridge = GLM4Bridge()
        bridge.hf_pretrained = omp_pretrained
        bridge.hf_config = omp_pretrained.config

        registry = bridge.mapping_registry()
        mapping = registry.megatron_to_hf_lookup("decoder.layers.0.mlp.linear_fc1.weight")

        assert isinstance(mapping, AutoMapping)
        assert mapping.hf_param == "model.layers.0.mlp.gate_up_proj.weight"
        assert not any(
            item.hf_param == "model.layers.*.self_attn.q_norm.weight"
            for item in registry.mappings
            if hasattr(item, "hf_param")
        )
        assert not any(
            item.hf_param == "model.layers.*.self_attn.k_norm.weight"
            for item in registry.mappings
            if hasattr(item, "hf_param")
        )
