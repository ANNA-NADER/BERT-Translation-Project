"""
Tests for BERT2BERT translation model.
"""

import pytest
import torch
from src.model.bert_translator import BERTTranslator


@pytest.fixture
def model_config():
    return {
        "model": {
            "name": "bert-base-uncased",
            "encoder": {"vocab_size": 30522},
            "generation": {"max_length": 20, "beam_size": 2},
        }
    }


@pytest.fixture
def model(model_config):
    return BERTTranslator(model_config)


def test_model_initialization(model):
    """Test that model initializes correctly."""
    assert isinstance(model, BERTTranslator)
    assert model.model is not None
    # Check config propagation
    # BERT CLS token ID is 101
    assert model.model.config.decoder_start_token_id == 101


def test_forward_pass(model):
    """Test forward pass with sample data."""
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    target_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    outputs = model(input_ids, target_ids, attention_mask)

    # Check output shape: (batch_size, seq_len, vocab_size)
    assert outputs.logits.shape == (batch_size, seq_len, 30522)


def test_generate(model):
    """Test generation method."""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))

    generated = model.generate(input_ids, max_length=15, num_beams=1)

    # Check that output is a tensor of token IDs
    assert isinstance(generated, torch.Tensor)
    assert generated.dim() == 2
    assert generated.size(0) == batch_size
    assert generated.size(1) <= 15
