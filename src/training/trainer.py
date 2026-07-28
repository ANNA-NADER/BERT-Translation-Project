"""
Training loop and utilities for BERT translation model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
from typing import Dict
from loguru import logger

from .metrics import (
    compute_bleu,
    compute_perplexity,
    compute_accuracy,
    decode_predictions,
    MetricsTracker,
)


class Trainer:
    """
    Trainer class for BERT translation model.
    Handles training loop, evaluation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        source_tokenizer,
        target_tokenizer,
        device: str = "cuda",
    ):
        """
        Args:
            model: Translation model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            source_tokenizer: Source language tokenizer
            target_tokenizer: Target language tokenizer
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config["training"]
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        # Device setup
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.use_amp = self.config.get("mixed_precision", False) and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_dir = Path(self.config["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        tensorboard_dir = Path(self.config["tensorboard_dir"])
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)

        # Logging configuration
        logger.add(self.log_dir / "train.log", rotation="500 MB", level="INFO")

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Weights & Biases setup
        self.use_wandb = self.config.get("use_wandb", False)
        if self.use_wandb:
            import wandb

            wandb.init(
                project=self.config.get("wandb_project", "bert-translation"),
                config=config,
                name=f"bert-translation-run-{self.global_step}",
            )

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {model.get_num_parameters():,}")
        logger.info(f"Mixed precision: {self.use_amp}")

    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_name = self.config.get("optimizer", "adamw").lower()

        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
                eps=self.config["adam_epsilon"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", "linear_warmup")

        if scheduler_name == "linear_warmup":
            num_training_steps = len(self.train_loader) * self.config["num_epochs"]
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config["warmup_steps"],
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config['num_epochs']}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with mixed precision
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with autocast(device_type=device_type, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    target_ids=target_ids,  # Pass full targets, HF handles shifting
                    attention_mask=attention_mask,
                )

                loss = outputs.loss
                logits = outputs.logits

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config["gradient_accumulation_steps"] == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["max_grad_norm"]
                )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            # Compute metrics
            with torch.no_grad():
                accuracy = compute_accuracy(logits, labels)
                perplexity = compute_perplexity(loss.item())

            # Update metrics
            batch_size = input_ids.size(0)
            metrics_tracker.update(
                {"loss": loss.item(), "perplexity": perplexity, "accuracy": accuracy},
                count=batch_size,
            )

            # Logging
            if self.global_step % self.config["logging_steps"] == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/perplexity", perplexity, self.global_step)
                self.writer.add_scalar("train/accuracy", accuracy, self.global_step)
                self.writer.add_scalar("train/learning_rate", lr, self.global_step)

                if self.use_wandb:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/perplexity": perplexity,
                            "train/accuracy": accuracy,
                            "train/learning_rate": lr,
                            "epoch": self.epoch + (batch_idx / len(self.train_loader)),
                        },
                        step=self.global_step,
                    )

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{perplexity:.2f}",
                    "acc": f"{accuracy:.2f}%",
                }
            )

            # Checkpointing
            if self.global_step % self.config["save_steps"] == 0:
                self.save_checkpoint(f"checkpoint-step-{self.global_step}")

            # Evaluation
            if self.global_step % self.config["eval_steps"] == 0:
                self.evaluate()
                self.model.train()  # Back to training mode

        return metrics_tracker.compute()

    @torch.no_grad()
    def evaluate(self, test_mode: bool = False) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            test_mode: If True, also compute BLEU score

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()

        all_predictions = []
        all_references = []
        all_sources = []

        loader = self.val_loader
        progress_bar = tqdm(loader, desc="Evaluating")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                target_ids=target_ids,
                attention_mask=attention_mask,
            )

            loss = outputs.loss
            logits = outputs.logits

            # Compute metrics
            accuracy = compute_accuracy(logits, labels)
            perplexity = compute_perplexity(loss.item())

            batch_size = input_ids.size(0)
            metrics_tracker.update(
                {"loss": loss.item(), "perplexity": perplexity, "accuracy": accuracy},
                count=batch_size,
            )

            # Generate translations for BLEU
            if test_mode:
                generated = self.model.generate(
                    input_ids,
                    max_length=self.config["max_target_length"],
                    num_beams=self.config.get("num_beams", 5),
                )

                predictions = decode_predictions(generated, self.target_tokenizer)
                references = decode_predictions(target_ids, self.target_tokenizer)
                sources = decode_predictions(input_ids, self.source_tokenizer)

                all_predictions.extend(predictions)
                all_references.extend(references)
                all_sources.extend(sources)

        # Compute average metrics
        metrics = metrics_tracker.compute()

        # Compute BLEU if in test mode
        if test_mode and all_predictions:
            bleu_metrics = compute_bleu(all_predictions, all_references)
            metrics.update(bleu_metrics)

        # Log to tensorboard
        for name, value in metrics.items():
            self.writer.add_scalar(f"val/{name}", value, self.global_step)

        # Log to wandb
        if self.use_wandb:
            import wandb

            wandb.log(
                {f"val/{name}": value for name, value in metrics.items()},
                step=self.global_step,
            )

            # Log translation samples table
            if test_mode and all_predictions:
                sample_table = wandb.Table(
                    columns=["Source (EN)", "Reference (FR)", "Prediction (FR)"]
                )
                for src, ref, pred in zip(
                    all_sources[:15], all_references[:15], all_predictions[:15]
                ):
                    sample_table.add_data(src, ref, pred)
                wandb.log({"evaluation_samples": sample_table}, step=self.global_step)

        logger.info(f"Validation metrics: {metrics}")

        return metrics

    def train(self):
        """
        Main training loop.
        """
        logger.info(f"Starting training for {self.config['num_epochs']} epochs...")
        logger.info(
            f"Total steps: {len(self.train_loader) * self.config['num_epochs']}"
        )

        for epoch in range(self.config["num_epochs"]):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            logger.info(f"Epoch {epoch + 1} training metrics:")
            for name, value in train_metrics.items():
                logger.info(f"  {name}: {value:.4f}")

            # Evaluate
            val_metrics = self.evaluate()

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model")
                logger.info(
                    f"New best model saved (val_loss: {self.best_val_loss:.4f})"
                )

            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")

        logger.info("Training completed!")
        if self.use_wandb:
            import wandb

            wandb.finish()
        self.writer.close()

    def save_checkpoint(self, name: str):
        """
        Save model checkpoint.

        Args:
            name: Checkpoint name
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        # Keep only last N checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        save_total_limit = self.config.get("save_total_limit", 3)

        # Get all step checkpoints
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-step-*.pt"),
            key=lambda x: int(x.stem.split("-")[-1]),
        )

        # Remove old checkpoints
        if len(checkpoints) > save_total_limit:
            for checkpoint in checkpoints[:-save_total_limit]:
                checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.epoch + 1}, step {self.global_step}")
