"""
Quick start training script for BERT translation
Trains on a small subset for testing
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.cli import train_model

if __name__ == "__main__":
    print("=" * 60)
    print("BERT Translation - Quick Start Training")
    print("=" * 60)
    print("\nThis will train on a small subset (1000 samples) for testing.")
    print("For full training, use: python app/cli.py train\n")

    # Train with small subset
    train_model(
        config_path="config/training_config.yaml",
        num_train_samples=1000,
        num_val_samples=200,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print(
        "1. Test translation: python app/cli.py translate "
        "--checkpoint checkpoints/best_model.pt --interactive"
    )
    print("2. Launch web UI: python app/web_interface.py --checkpoint checkpoints/best_model.pt")
    print("3. Start API: python app/api.py --checkpoint checkpoints/best_model.pt")
