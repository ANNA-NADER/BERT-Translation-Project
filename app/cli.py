"""
Command-line interface for BERT translation
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_dataloaders
from src.model import BERTTranslator
from src.training import Trainer
from src.utils import get_tokenizers


def translate_interactive(model, source_tokenizer, target_tokenizer, device):
    """
    Interactive translation mode.
    """
    print("\n" + "=" * 50)
    print("BERT Translation - Interactive Mode")
    print("=" * 50)
    print("Enter text to translate (or 'quit' to exit)")
    print()

    model.eval()

    while True:
        try:
            # Get input
            print("English: ", end="", flush=True)
            text = input().strip()

            if text.lower() in ["quit", "exit", "q"]:
                break

            if not text:
                continue

            # Tokenize
            inputs = source_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=128
            )

            input_ids = inputs["input_ids"].to(device)

            # Generate translation
            with torch.no_grad():
                generated = model.generate(input_ids, max_length=128, num_beams=5)

            # Decode
            translation = target_tokenizer.decode(generated[0], skip_special_tokens=True)

            print(f"French:  {translation}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

    print("\nGoodbye!")


def translate_file(input_file, output_file, model, source_tokenizer, target_tokenizer, device):
    """
    Translate a file line by line.
    """
    print(f"Translating {input_file} -> {output_file}")

    model.eval()

    with (
        open(input_file, encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()

            if not line:
                f_out.write("\n")
                continue

            # Tokenize
            inputs = source_tokenizer(
                line, return_tensors="pt", padding=True, truncation=True, max_length=128
            )

            input_ids = inputs["input_ids"].to(device)

            # Generate translation
            with torch.no_grad():
                generated = model.generate(input_ids, max_length=128, num_beams=5)

            # Decode
            translation = target_tokenizer.decode(generated[0], skip_special_tokens=True)

            f_out.write(translation + "\n")

            if line_num % 100 == 0:
                print(f"Translated {line_num} lines...")

    print(f"Translation complete! Output saved to {output_file}")


def train_model(config_path, num_train_samples=None, num_val_samples=None):
    """
    Train the translation model.
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set device
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Load tokenizers
    source_tokenizer, target_tokenizer = get_tokenizers()

    # Update model config with actual vocab sizes
    model_config_path = Path(config_path).parent / "model_config.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    model_config["model"]["vocab"]["target_vocab_size"] = len(target_tokenizer)

    # Create model
    print("\nInitializing model...")
    model = BERTTranslator(model_config)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, _test_loader = create_dataloaders(
        config,
        source_tokenizer,
        target_tokenizer,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        device=device,
    )

    # Train
    trainer.train()

    print("\nTraining completed!")


def evaluate_model(checkpoint_path, config_path):
    """
    Evaluate a trained model.
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizers
    source_tokenizer, target_tokenizer = get_tokenizers()

    # Load model config
    model_config_path = Path(config_path).parent / "model_config.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    model_config["model"]["vocab"]["target_vocab_size"] = len(target_tokenizer)

    # Create model
    model = BERTTranslator(model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")

    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        config,
        source_tokenizer,
        target_tokenizer,
        num_train_samples=100,
        num_val_samples=100,
    )

    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        train_loader=test_loader,  # Dummy
        val_loader=test_loader,
        config=config,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        device=device,
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(test_mode=True)

    print("\nTest Results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="BERT Translation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training config",
    )
    train_parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Limit training samples (for testing)",
    )
    train_parser.add_argument(
        "--num-val-samples",
        type=int,
        default=None,
        help="Limit validation samples (for testing)",
    )

    # Translate command
    translate_parser = subparsers.add_parser("translate", help="Translate text")
    translate_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    translate_parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to config",
    )
    translate_parser.add_argument("--input", type=str, help="Input file to translate")
    translate_parser.add_argument("--output", type=str, help="Output file for translations")
    translate_parser.add_argument(
        "--interactive", action="store_true", help="Interactive translation mode"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to config",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            args.config,
            num_train_samples=args.num_train_samples,
            num_val_samples=args.num_val_samples,
        )

    elif args.command == "translate":
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        source_tokenizer, target_tokenizer = get_tokenizers()

        model_config_path = Path(args.config).parent / "model_config.yaml"
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)

        model_config["model"]["vocab"]["target_vocab_size"] = len(target_tokenizer)

        model = BERTTranslator(model_config)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        if args.interactive:
            translate_interactive(model, source_tokenizer, target_tokenizer, device)
        elif args.input and args.output:
            translate_file(
                args.input,
                args.output,
                model,
                source_tokenizer,
                target_tokenizer,
                device,
            )
        else:
            print("Error: Specify --interactive or both --input and --output")

    elif args.command == "evaluate":
        evaluate_model(args.checkpoint, args.config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
