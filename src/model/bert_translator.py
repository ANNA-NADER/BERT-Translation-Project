"""
BERT-based Neural Machine Translation Model
Combines BERT encoder with transformer decoder for translation
"""

import torch
import torch.nn as nn
from transformers import EncoderDecoderModel
from typing import Optional, Dict
import yaml


class BERTTranslator(nn.Module):
    """
    BERT-based sequence-to-sequence translation model using Hugging Face EncoderDecoderModel.
    Uses pre-trained BERT for both encoder and decoder (BERT2BERT).
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        model_config = config["model"]

        # Load pre-trained BERT2BERT model
        # Using bert-base-uncased for both encoder and decoder
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_config["name"], model_config["name"]
        )

        # Configure model parameters
        # Configure model parameters
        self.model.config.vocab_size = self.model.config.encoder.vocab_size
        self.model.config.decoder_start_token_id = getattr(
            self.model.config.encoder, "cls_token_id", 101
        )
        self.model.config.pad_token_id = getattr(
            self.model.config.encoder, "pad_token_id", 0
        )
        self.model.config.eos_token_id = getattr(
            self.model.config.encoder, "sep_token_id", 102
        )

        # Generation config
        self.gen_config = model_config.get("generation", {})
        self.max_length = self.gen_config.get("max_length", 128)
        self.beam_size = self.gen_config.get("beam_size", 5)

        # Set generation parameters using GenerationConfig to avoid deprecation warnings
        from transformers import GenerationConfig

        self.model.generation_config = GenerationConfig(
            max_length=self.max_length,
            num_beams=self.beam_size,
            early_stopping=True,
            no_repeat_ngram_size=3,
            bos_token_id=self.model.config.decoder_start_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
        )

    @classmethod
    def from_config_file(cls, config_path: str):
        """Load model from configuration file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            input_ids: Source token IDs
            target_ids: Target token IDs
            attention_mask: Source attention mask
            target_mask: Unused (HF handles this internally)

        Returns:
            ModelOutput with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=target_ids
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Generate translation.

        Args:
            input_ids: Source token IDs
            max_length: Maximum generation length
            num_beams: Number of beams
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling for generation

        Returns:
            Generated token IDs
        """
        # Build generator arguments dynamically
        gen_kwargs = {
            "input_ids": input_ids,
            "decoder_start_token_id": self.model.config.decoder_start_token_id,
            "max_length": max_length or self.max_length,
            "num_beams": num_beams or self.beam_size,
        }

        # Only pass sampling arguments when do_sample is True to prevent Hugging Face warnings
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p

        # Use HF model generation
        generated_ids = self.model.generate(**gen_kwargs)

        return generated_ids

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
