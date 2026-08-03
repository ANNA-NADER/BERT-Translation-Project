"""
Tokenizer utilities for translation
"""

from pathlib import Path

from transformers import AutoTokenizer, BertTokenizer, PreTrainedTokenizer


def get_tokenizers(
    source_model: str = "bert-base-uncased",
    target_model: str = "bert-base-uncased",
    cache_dir: str | None = None,
) -> tuple[PreTrainedTokenizer, PreTrainedTokenizer]:
    """
    Load source and target tokenizers.

    Args:
        source_model: Source language model name or path
        target_model: Target language model name or path
        cache_dir: Cache directory for tokenizers

    Returns:
        Tuple of (source_tokenizer, target_tokenizer)
    """
    print(f"Loading source tokenizer: {source_model}")
    source_tokenizer = BertTokenizer.from_pretrained(source_model, cache_dir=cache_dir)

    print(f"Loading target tokenizer: {target_model}")
    target_tokenizer = AutoTokenizer.from_pretrained(target_model, cache_dir=cache_dir)

    # Ensure special tokens are set
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    if target_tokenizer.bos_token is None:
        target_tokenizer.bos_token = target_tokenizer.eos_token

    print(f"Source vocab size: {len(source_tokenizer)}")
    print(f"Target vocab size: {len(target_tokenizer)}")

    return source_tokenizer, target_tokenizer


def save_tokenizers(
    source_tokenizer: PreTrainedTokenizer,
    target_tokenizer: PreTrainedTokenizer,
    save_dir: str,
):
    """
    Save tokenizers to directory.

    Args:
        source_tokenizer: Source language tokenizer
        target_tokenizer: Target language tokenizer
        save_dir: Directory to save tokenizers
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    source_path = save_path / "source_tokenizer"
    target_path = save_path / "target_tokenizer"

    source_tokenizer.save_pretrained(source_path)
    target_tokenizer.save_pretrained(target_path)

    print(f"Tokenizers saved to {save_dir}")


def load_tokenizers(load_dir: str) -> tuple[PreTrainedTokenizer, PreTrainedTokenizer]:
    """
    Load tokenizers from directory.

    Args:
        load_dir: Directory containing saved tokenizers

    Returns:
        Tuple of (source_tokenizer, target_tokenizer)
    """
    load_path = Path(load_dir)

    source_path = load_path / "source_tokenizer"
    target_path = load_path / "target_tokenizer"

    source_tokenizer = BertTokenizer.from_pretrained(source_path)
    target_tokenizer = AutoTokenizer.from_pretrained(target_path)

    print(f"Tokenizers loaded from {load_dir}")

    return source_tokenizer, target_tokenizer


if __name__ == "__main__":
    # Test tokenizer loading
    source_tok, target_tok = get_tokenizers()

    # Test encoding
    source_text = "Hello, how are you?"
    target_text = "Bonjour, comment allez-vous?"

    source_ids = source_tok.encode(source_text)
    target_ids = target_tok.encode(target_text)

    print(f"\nSource: {source_text}")
    print(f"Encoded: {source_ids}")
    print(f"Decoded: {source_tok.decode(source_ids)}")

    print(f"\nTarget: {target_text}")
    print(f"Encoded: {target_ids}")
    print(f"Decoded: {target_tok.decode(target_ids)}")
