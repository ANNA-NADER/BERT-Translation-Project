# Iterative Training Guide

## Training in Stages and Resuming from Checkpoints

The training process can be run incrementally. This allows you to:
- Evaluate model quality at intermediate stages.
- Halt training early if performance reaches an acceptable level.
- Fit training sessions into smaller time blocks.

---

## Incremental Training Strategy

### Stage 1: Initial Test (5,000 samples)
Run a quick training run on a small subset of the data:
```bash
python app/cli.py train --config config/training_config.yaml --num-train-samples 5000 --num-val-samples 500
```

Verify translation functionality:
```bash
python app/cli.py translate --checkpoint checkpoints/best_model.pt --interactive
```

---

### Stage 2: Small Dataset (20,000 samples)
Before starting, update the configuration to resume training from the previous checkpoint.

Edit `config/training_config.yaml`:
```yaml
training:
  # ...
  resume_from_checkpoint: "checkpoints/best_model.pt"
```

Then run training with more samples:
```bash
python app/cli.py train --config config/training_config.yaml --num-train-samples 20000 --num-val-samples 2000
```

Test the updated model:
```bash
python app/cli.py translate --checkpoint checkpoints/best_model.pt --interactive
```

---

### Stage 3: Medium Dataset (50,000 samples)
Increase training coverage:
```bash
python app/cli.py train --config config/training_config.yaml --num-train-samples 50000 --num-val-samples 5000
```

---

### Stage 4: Full Dataset Training
When ready to train on the entire dataset:
```bash
python app/cli.py train --config config/training_config.yaml
```

---

## Alternative: Adjusting Epochs

Instead of increasing dataset size, you can train on the same data subset for additional epochs:

1. **Set initial epochs** in `config/training_config.yaml`:
   ```yaml
   training:
     num_epochs: 5
   ```
   Run training:
   ```bash
   python app/cli.py train --config config/training_config.yaml --num-train-samples 10000
   ```

2. **Increase the target epoch count** and resume:
   Edit `config/training_config.yaml`:
   ```yaml
   training:
     num_epochs: 10
     resume_from_checkpoint: "checkpoints/best_model.pt"
   ```
   Run training:
   ```bash
   python app/cli.py train --config config/training_config.yaml --num-train-samples 10000
   ```

---

## Monitoring Progress

Check the performance metrics and test translations at each stage:
```bash
# Get evaluation metrics
python app/cli.py evaluate --checkpoint checkpoints/best_model.pt

# Test translation output
python app/cli.py translate --checkpoint checkpoints/best_model.pt --interactive
```

---

## Recommended Training Path

| Stage | Samples | Est. Time | Expected BLEU | Action |
|-------|---------|-----------|---------------|--------|
| 1 | 5,000 | 5 min | 10-15 | Verify pipeline |
| 2 | 20,000 | 20 min | 15-20 | Assess improvement |
| 3 | 50,000 | 1 hr | 20-25 | Check performance |
| 4 | 100,000 | 2 hr | 25-30 | Final adjustments |
| 5 | Full | 6 hr | 30-35 | Production release |

---

## Key Points

1. Checkpoints are automatically written to the `checkpoints/` directory.
2. The best-performing model checkpoint is tracked as `checkpoints/best_model.pt`.
3. Training can be interrupted via `Ctrl+C` without losing the current checkpoint.
4. Set the `resume_from_checkpoint` config option to load model state when starting a run.
