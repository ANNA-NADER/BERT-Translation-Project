"""
Unit tests for data preprocessing and dataset
"""

import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer, BertTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import TranslationDataset
from src.data.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None

    def test_normalize_text(self):
        """Test text normalization."""
        preprocessor = TextPreprocessor(lowercase=True)
        text = "Hello World"
        normalized = preprocessor.normalize_text(text)
        assert normalized == "hello world"

    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = TextPreprocessor()
        text = "Hello   World  \n\n  Test"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "Hello World Test"

    def test_preprocess(self):
        """Test full preprocessing pipeline."""
        preprocessor = TextPreprocessor(lowercase=True)
        text = "  Hello   World!  "
        processed = preprocessor.preprocess(text)
        assert processed == "hello world!"

    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        preprocessor = TextPreprocessor(lowercase=True)
        texts = ["Hello World", "Test Text", "Another Example"]
        processed = preprocessor.preprocess_batch(texts)
        assert len(processed) == 3
        assert processed[0] == "hello world"

    def test_filter_by_length(self):
        """Test length filtering."""
        source_texts = ["short sentence", "this is a medium length sentence", "x"]
        target_texts = ["phrase courte", "ceci est une phrase de longueur moyenne", "y"]

        filtered_source, _filtered_target = TextPreprocessor.filter_by_length(
            source_texts, target_texts, min_length=2, max_length=10
        )

        assert len(filtered_source) == 2
        assert "x" not in filtered_source

    def test_remove_duplicates(self):
        """Test duplicate removal."""
        source_texts = ["hello", "world", "hello"]
        target_texts = ["bonjour", "monde", "bonjour"]

        unique_source, _unique_target = TextPreprocessor.remove_duplicates(
            source_texts, target_texts
        )

        assert len(unique_source) == 2
        assert unique_source == ["hello", "world"]


class TestDataset:
    """Tests for TranslationDataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        source_texts = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I love machine learning.",
        ]
        target_texts = [
            "Bonjour, comment allez-vous?",
            "Le temps est beau aujourd'hui.",
            "J'aime l'apprentissage automatique.",
        ]
        return source_texts, target_texts

    def test_dataset_length(self, sample_data):
        """Test dataset length."""
        source_texts, target_texts = sample_data

        source_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        target_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

        dataset = TranslationDataset(source_texts, target_texts, source_tokenizer, target_tokenizer)

        assert len(dataset) == 3

    def test_dataset_getitem(self, sample_data):
        """Test dataset item retrieval."""
        source_texts, target_texts = sample_data

        source_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        target_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

        dataset = TranslationDataset(
            source_texts,
            target_texts,
            source_tokenizer,
            target_tokenizer,
            max_source_length=32,
            max_target_length=32,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "target_ids" in item
        assert "labels" in item

        assert item["input_ids"].shape == (32,)
        assert item["target_ids"].shape == (32,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
