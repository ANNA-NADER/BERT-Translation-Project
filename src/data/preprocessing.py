"""
Text preprocessing utilities for translation
"""

import re
import unicodedata
from typing import List


class TextPreprocessor:
    """
    Text preprocessing for neural machine translation.
    Handles normalization, cleaning, and basic tokenization.
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_accents: bool = False,
        normalize_unicode: bool = True,
    ):
        """
        Args:
            lowercase: Whether to lowercase text
            remove_accents: Whether to remove accents
            normalize_unicode: Whether to normalize unicode characters
        """
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.normalize_unicode = normalize_unicode

    def normalize_text(self, text: str) -> str:
        """
        Normalize text using various techniques.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if self.normalize_unicode:
            # Normalize unicode to NFC form
            text = unicodedata.normalize("NFC", text)

        if self.remove_accents:
            # Remove accents
            text = "".join(
                c
                for c in unicodedata.normalize("NFD", text)
                if unicodedata.category(c) != "Mn"
            )

        if self.lowercase:
            text = text.lower()

        return text

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and normalizing whitespace.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove control characters
        text = "".join(
            ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
        )

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        text = self.normalize_text(text)
        text = self.clean_text(text)
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]

    @staticmethod
    def filter_by_length(
        source_texts: List[str],
        target_texts: List[str],
        min_length: int = 1,
        max_length: int = 200,
        source_tokenizer=None,
        target_tokenizer=None,
    ) -> tuple:
        """
        Filter parallel texts by length.

        Args:
            source_texts: Source language texts
            target_texts: Target language texts
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            source_tokenizer: Tokenizer for source (if None, uses word count)
            target_tokenizer: Tokenizer for target (if None, uses word count)

        Returns:
            Filtered source and target texts
        """
        filtered_source = []
        filtered_target = []

        for src, tgt in zip(source_texts, target_texts):
            # Get lengths
            if source_tokenizer:
                src_len = len(source_tokenizer.encode(src))
            else:
                src_len = len(src.split())

            if target_tokenizer:
                tgt_len = len(target_tokenizer.encode(tgt))
            else:
                tgt_len = len(tgt.split())

            # Filter by length
            if (
                min_length <= src_len <= max_length
                and min_length <= tgt_len <= max_length
            ):
                filtered_source.append(src)
                filtered_target.append(tgt)

        return filtered_source, filtered_target

    @staticmethod
    def remove_duplicates(source_texts: List[str], target_texts: List[str]) -> tuple:
        """
        Remove duplicate sentence pairs.

        Args:
            source_texts: Source language texts
            target_texts: Target language texts

        Returns:
            Deduplicated source and target texts
        """
        seen = set()
        unique_source = []
        unique_target = []

        for src, tgt in zip(source_texts, target_texts):
            pair = (src, tgt)
            if pair not in seen:
                seen.add(pair)
                unique_source.append(src)
                unique_target.append(tgt)

        return unique_source, unique_target
