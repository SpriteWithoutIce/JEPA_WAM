"""Qwen2.5 LLM backbone with LoRA."""

from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMWrapper(nn.Module):
    """Thin wrapper around a HuggingFace causal LM with optional LoRA.

    Provides a unified interface for:
      - `get_input_embeddings()` -> nn.Embedding
      - `forward(inputs_embeds=..., output_hidden_states=...)`
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.llm.model_name
        self.d_llm = cfg.llm.d_llm

        # Load base model in full precision
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze backbone
        for p in self.llm.parameters():
            p.requires_grad = False

        # Apply LoRA
        lora_config = LoraConfig(
            r=cfg.llm.lora_rank,
            lora_alpha=cfg.llm.lora_alpha,
            target_modules=cfg.llm.lora_targets,
            lora_dropout=cfg.llm.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.llm.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        use_cache: bool = False,
    ):
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=True,
        )
