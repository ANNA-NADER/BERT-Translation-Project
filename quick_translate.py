"""
Bidirectional English <-> French CLI translator using Helsinki-NLP MarianMT.
Compatible with transformers v5.x.
"""

import torch
from transformers import MarianMTModel, MarianTokenizer

MODELS = {
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
}

LABELS = {
    "en-fr": ("English", "French"),
    "fr-en": ("French", "English"),
}


class Translator:
    def __init__(self, model_name: str):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.eval()

    def translate(self, text: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=5,
                max_length=max_length,
            )
        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )


def main():
    print("=" * 60)
    print("  English <-> French Translator  (Helsinki-NLP MarianMT)")
    print("=" * 60)
    print("\nLoading models... (first run may download weights)\n")

    translators = {}
    for key, model_name in MODELS.items():
        src, tgt = LABELS[key]
        print(f"  Loading {src} -> {tgt} model...")
        translators[key] = Translator(model_name)

    direction = "en-fr"
    print("\nBoth models loaded.")
    print("Commands: 'swap' to switch direction, 'quit' to exit.\n")
    print(f"Direction: {LABELS[direction][0]} -> {LABELS[direction][1]}\n")

    while True:
        try:
            src_label, tgt_label = LABELS[direction]
            print(f"{src_label}: ", end="", flush=True)
            text = input().strip()

            if text.lower() in ["quit", "exit", "q"]:
                break

            if text.lower() == "swap":
                direction = "fr-en" if direction == "en-fr" else "en-fr"
                src_label, tgt_label = LABELS[direction]
                print(f"\nSwitched to: {src_label} -> {tgt_label}\n")
                continue

            if not text:
                continue

            result = translators[direction].translate(text)
            print(f"{tgt_label}: {result}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
