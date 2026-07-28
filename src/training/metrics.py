"""
Evaluation metrics for translation
"""

import torch
from sacrebleu import corpus_bleu
from typing import List, Dict
import math


def compute_bleu(
    predictions: List[str], references: List[str], tokenize: str = "13a"
) -> Dict[str, float]:
    """
    Compute BLEU score using SacreBLEU.

    Args:
        predictions: List of predicted translations
        references: List of reference translations
        tokenize: Tokenization method ('13a', 'intl', 'zh', 'ja-mecab')

    Returns:
        Dictionary with BLEU score and related metrics
    """
    # SacreBLEU expects references as list of lists
    references = [[ref] for ref in references]

    # Compute BLEU
    bleu = corpus_bleu(predictions, references, tokenize=tokenize)

    return {
        "bleu": bleu.score,
        "bleu_1": bleu.precisions[0],
        "bleu_2": bleu.precisions[1],
        "bleu_3": bleu.precisions[2],
        "bleu_4": bleu.precisions[3],
        "bp": bleu.bp,  # Brevity penalty
    }


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss

    Returns:
        Perplexity value
    """
    return math.exp(min(loss, 100))  # Cap to prevent overflow


def compute_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy.

    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        labels: Ground truth labels (batch_size, seq_len)
        ignore_index: Index to ignore in accuracy computation

    Returns:
        Accuracy percentage
    """
    predictions = torch.argmax(logits, dim=-1)

    # Create mask for valid tokens
    mask = labels != ignore_index

    # Compute accuracy
    correct = (predictions == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item()

    return accuracy * 100


def decode_predictions(
    token_ids: torch.Tensor, tokenizer, skip_special_tokens: bool = True
) -> List[str]:
    """
    Decode token IDs to text.

    Args:
        token_ids: Tensor of token IDs (batch_size, seq_len)
        tokenizer: Tokenizer for decoding
        skip_special_tokens: Whether to skip special tokens

    Returns:
        List of decoded texts
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().numpy()

    texts = []
    for ids in token_ids:
        text = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        texts.append(text)

    return texts


class MetricsTracker:
    """
    Track and compute running averages of metrics during training.
    """

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metric_dict: Dict[str, float], count: int = 1):
        """
        Update metrics with new values.

        Args:
            metric_dict: Dictionary of metric names and values
            count: Number of samples (for weighted average)
        """
        for name, value in metric_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0

            self.metrics[name] += value * count
            self.counts[name] += count

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.

        Returns:
            Dictionary of averaged metrics
        """
        return {name: self.metrics[name] / self.counts[name] for name in self.metrics}

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def __repr__(self) -> str:
        metrics = self.compute()
        return ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
