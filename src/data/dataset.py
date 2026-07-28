"""
Dataset classes for translation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path

from .preprocessing import TextPreprocessor


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel translation data.
    """

    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_tokenizer,
        target_tokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        """
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts
            source_tokenizer: Tokenizer for source language
            target_tokenizer: Tokenizer for target language
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            preprocessor: Text preprocessor
        """
        assert len(source_texts) == len(target_texts), (
            "Source and target must have same length"
        )

        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.preprocessor = preprocessor or TextPreprocessor()

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary with input_ids, attention_mask, target_ids, and labels
        """
        # Get texts
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]

        # Preprocess
        source_text = self.preprocessor.preprocess(source_text)
        target_text = self.preprocessor.preprocess(target_text)

        # Tokenize source
        source_encoding = self.source_tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.target_tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare decoder input (shift right)
        target_ids = target_encoding["input_ids"].squeeze(0)

        # Labels for loss computation (ignore padding)
        labels = target_ids.clone()
        labels[labels == self.target_tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "target_ids": target_ids,
            "labels": labels,
        }


def load_opus100_dataset(
    language_pair: str = "en-fr",
    split: str = "train",
    cache_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Load OPUS-100 dataset from Hugging Face.

    Args:
        language_pair: Language pair (e.g., "en-fr")
        split: Dataset split ("train", "validation", "test")
        cache_dir: Cache directory for datasets
        num_samples: Number of samples to load (None = all)

    Returns:
        Tuple of (source_texts, target_texts)
    """
    print(f"Loading OPUS-100 dataset ({language_pair}, {split})...")

    # Load dataset
    dataset = load_dataset(
        "Helsinki-NLP/opus-100", language_pair, split=split, cache_dir=cache_dir
    )

    # Limit samples if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Extract parallel texts
    source_lang, target_lang = language_pair.split("-")
    source_texts = [item["translation"][source_lang] for item in dataset]
    target_texts = [item["translation"][target_lang] for item in dataset]

    print(f"Loaded {len(source_texts)} parallel sentences")

    return source_texts, target_texts


def create_dataloaders(
    config: Dict,
    source_tokenizer,
    target_tokenizer,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Training configuration
        source_tokenizer: Source language tokenizer
        target_tokenizer: Target language tokenizer
        num_train_samples: Limit training samples (for testing)
        num_val_samples: Limit validation samples (for testing)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_config = config["training"]

    # Load datasets
    print("Loading training data...")
    train_source, train_target = load_opus100_dataset(
        language_pair=train_config["dataset_config"],
        split=train_config["train_split"],
        num_samples=num_train_samples,
    )

    print("Loading validation data...")
    val_source, val_target = load_opus100_dataset(
        language_pair=train_config["dataset_config"],
        split=train_config["val_split"],
        num_samples=num_val_samples,
    )

    print("Loading test data...")
    test_source, test_target = load_opus100_dataset(
        language_pair=train_config["dataset_config"],
        split=train_config["test_split"],
        num_samples=num_val_samples,
    )

    # Create preprocessor
    preprocessor = TextPreprocessor(
        lowercase=False, remove_accents=False, normalize_unicode=True
    )

    # Filter by length
    print("Filtering by length...")
    train_source, train_target = TextPreprocessor.filter_by_length(
        train_source,
        train_target,
        min_length=train_config["min_length"],
        max_length=train_config["max_length"],
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    val_source, val_target = TextPreprocessor.filter_by_length(
        val_source,
        val_target,
        min_length=train_config["min_length"],
        max_length=train_config["max_length"],
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    print(
        f"After filtering: {len(train_source)} train, {len(val_source)} val, {len(test_source)} test"
    )

    # Create datasets
    train_dataset = TranslationDataset(
        train_source,
        train_target,
        source_tokenizer,
        target_tokenizer,
        max_source_length=train_config["max_source_length"],
        max_target_length=train_config["max_target_length"],
        preprocessor=preprocessor,
    )

    val_dataset = TranslationDataset(
        val_source,
        val_target,
        source_tokenizer,
        target_tokenizer,
        max_source_length=train_config["max_source_length"],
        max_target_length=train_config["max_target_length"],
        preprocessor=preprocessor,
    )

    test_dataset = TranslationDataset(
        test_source,
        test_target,
        source_tokenizer,
        target_tokenizer,
        max_source_length=train_config["max_source_length"],
        max_target_length=train_config["max_target_length"],
        preprocessor=preprocessor,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=True if train_config["device"] == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["eval_batch_size"],
        shuffle=False,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=True if train_config["device"] == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["eval_batch_size"],
        shuffle=False,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=True if train_config["device"] == "cuda" else False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")

    config_path = (
        Path(__file__).parent.parent.parent / "config" / "training_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizers
    source_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    target_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        source_tokenizer,
        target_tokenizer,
        num_train_samples=100,
        num_val_samples=50,
    )

    print("\nDataset sizes:")
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val: {len(val_loader.dataset)}")
    print(f"Test: {len(test_loader.dataset)}")

    # Test batch
    batch = next(iter(train_loader))
    print("\nBatch shapes:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
