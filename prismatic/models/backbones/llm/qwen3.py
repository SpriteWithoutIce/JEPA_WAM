"""
qwen3.py

Class definition for all LLMs derived from Qwen3ForCausalLM.
"""

from typing import Optional, Sequence, Type

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder
from prismatic.models.backbones.llm.prompting.qwen_prompter import QwenPromptBuilder

# Registry =>> Support Qwen-3 Models (from HF Transformers)
# fmt: off
QWEN3_MODELS = {
    # === Pure Qwen3 (non-instruct/chat-tuned) Models ===
    "qwen3-1_7b-pure": {
        "llm_family": "qwen3", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen3-1.7B-Base"
    },
}
# fmt: on


class Qwen3LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        llm_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_path=llm_path,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **QWEN3_MODELS[llm_backbone_id],
        )

        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return QwenPromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[torch.nn.Module]:
        return Qwen3DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[torch.nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)
