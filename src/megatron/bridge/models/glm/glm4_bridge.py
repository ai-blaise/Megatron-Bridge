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

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(source="Glm4ForCausalLM", target=GPTModel, model_type="glm4")
class GLM4Bridge(MegatronModelBridge):
    """Megatron Bridge for GLM-4 causal language models.

    Handles the dense GLM-4 layout and the OMP variant that uses fused
    ``gate_up_proj`` MLP weights and omits explicit Q/K layernorm tensors.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace GLM-4 config to Megatron GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = True
        provider.qk_layernorm = self._uses_qk_layernorm(hf_pretrained)
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.position_embedding_type = "rope"
        provider.rotary_percent = 1.0
        provider.share_embeddings_and_output_weights = getattr(hf_pretrained.config, "tie_word_embeddings", False)
        provider.autocast_dtype = torch.bfloat16
        provider.gradient_accumulation_fusion = False

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return GLM-4 HF <-> Megatron parameter mappings."""
        use_fused_gate_up_proj = self._uses_fused_gate_up_proj()
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.layers.*.input_layernorm.weight"
            ),
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]

        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
            ]
        )
        if use_fused_gate_up_proj:
            mapping_list.append(
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    hf_param="model.layers.*.mlp.gate_up_proj.weight",
                )
            )
        else:
            mapping_list.append(
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                )
            )

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        """Dequantize FP8 GLM-4 weights when sibling scale tensors are present."""
        return quantization_utils.maybe_dequantize_hf_quantized_weight(hf_param, hf_state_dict)

    def _hf_state_source(self, hf_pretrained: PreTrainedCausalLM | None = None):
        """Return the HF state source when the model has been loaded."""
        if hf_pretrained is None:
            hf_pretrained = getattr(self, "hf_pretrained", None)
        hf_state = getattr(hf_pretrained, "state", None)
        if hf_state is None:
            return None
        return getattr(hf_state, "source", None)

    def _hf_has_glob(self, pattern: str, hf_pretrained: PreTrainedCausalLM | None = None) -> bool:
        """Check whether the loaded HF checkpoint exposes keys matching *pattern*."""
        hf_source = self._hf_state_source(hf_pretrained)
        return bool(hf_source is not None and hasattr(hf_source, "has_glob") and hf_source.has_glob(pattern))

    def _uses_fused_gate_up_proj(self, hf_pretrained: PreTrainedCausalLM | None = None) -> bool:
        """Detect the OMP fused MLP layout."""
        return self._hf_has_glob("*mlp.gate_up_proj.weight*", hf_pretrained)

    def _uses_qk_layernorm(self, hf_pretrained: PreTrainedCausalLM | None = None) -> bool:
        """Detect whether the HF checkpoint stores explicit Q/K layernorm tensors."""
        config = getattr(hf_pretrained, "config", hf_pretrained)
        default_value = bool(getattr(config, "use_qk_norm", True))
        hf_source = self._hf_state_source(hf_pretrained)
        if hf_source is None or not hasattr(hf_source, "has_glob"):
            return default_value
        return self._hf_has_glob("*self_attn.q_norm.weight*", hf_pretrained) and self._hf_has_glob(
            "*self_attn.k_norm.weight*",
            hf_pretrained,
        )
