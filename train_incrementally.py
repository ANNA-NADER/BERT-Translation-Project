"""
Incremental training script - train in stages
"""

import subprocess
import sys
from pathlib import Path


def run_training(num_samples, num_val_samples, stage_name):
    """Run training for a specific stage."""
    print("\n" + "=" * 70)
    print(f"STAGE: {stage_name}")
    print(f"Training samples: {num_samples:,}")
    print(f"Validation samples: {num_val_samples:,}")
    print("=" * 70 + "\n")

    cmd = [
        sys.executable,
        "app/cli.py",
        "train",
        "--config",
        "config/training_config.yaml",
        "--num-train-samples",
        str(num_samples),
        "--num-val-samples",
        str(num_val_samples),
    ]

    subprocess.run(cmd)


def test_model():
    """Prompt user to test the model."""
    print("\n" + "=" * 70)
    print("TESTING TIME!")
    print("=" * 70)
    print("\nTest your model with:")
    print("  python app/cli.py translate --checkpoint checkpoints/best_model.pt --interactive")
    print("\nOr evaluate metrics:")
    print("  python app/cli.py evaluate --checkpoint checkpoints/best_model.pt")
    print("\n" + "=" * 70)

    response = input("\nContinue to next stage? (y/n): ").strip().lower()
    return response == "y"


def main():
    print("=" * 70)
    print("ITERATIVE TRAINING - Train in Stages")
    print("=" * 70)
    print("\nThis will train your model incrementally:")
    print("  Stage 1: 5,000 samples (~5 minutes)")
    print("  Stage 2: 20,000 samples (~20 minutes)")
    print("  Stage 3: 50,000 samples (~1 hour)")
    print("  Stage 4: 100,000 samples (~2 hours)")
    print("\nYou can stop at any stage if quality is good enough!")
    print("=" * 70)

    input("\nPress Enter to start Stage 1...")

    # Stage 1: 5,000 samples
    run_training(5000, 500, "Stage 1 - Quick Test (5K samples)")
    if not test_model():
        print("\nTraining stopped. Resume anytime by running this script again.")
        return

    # Update config to resume from checkpoint
    print("\nUpdating config to resume from checkpoint...")
    config_path = Path("config/training_config.yaml")
    config_text = config_path.read_text()

    if "resume_from_checkpoint:" not in config_text:
        # Add resume_from_checkpoint setting
        config_text = config_text.replace(
            "resume_from_checkpoint: null",
            'resume_from_checkpoint: "checkpoints/best_model.pt"',
        )
        config_path.write_text(config_text)
        print("✓ Config updated to resume from checkpoint")

    # Stage 2: 20,000 samples
    run_training(20000, 2000, "Stage 2 - Small Dataset (20K samples)")
    if not test_model():
        print("\nTraining stopped. Resume anytime by running this script again.")
        return

    # Stage 3: 50,000 samples
    run_training(50000, 5000, "Stage 3 - Medium Dataset (50K samples)")
    if not test_model():
        print("\nTraining stopped. Resume anytime by running this script again.")
        return

    # Stage 4: 100,000 samples
    run_training(100000, 10000, "Stage 4 - Large Dataset (100K samples)")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYour model is ready! Test it with:")
    print("  python app/cli.py translate --checkpoint checkpoints/best_model.pt --interactive")
    print("\nFor even better quality, run full training:")
    print("  python app/cli.py train --config config/training_config.yaml")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Your progress is saved in checkpoints/")
        print("Resume by running this script again.")
