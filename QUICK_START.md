# BERT Translation - Quick Start

This document provides quick instructions to test and run the trained BERT-based translator model.

The best performing model checkpoint is saved at:
`checkpoints/best_model.pt`

---

## Next Steps

### 1. Interactive Translation
Run the translation system in interactive command-line mode:
```bash
python app/cli.py translate --checkpoint checkpoints/best_model.pt --interactive
```
- Input English sentences at the prompt.
- Enter 'quit' or 'exit' to close the interactive session.

---

### 2. Web Interface
Launch the local Gradio-based web user interface:
```bash
python app/web_interface.py --checkpoint checkpoints/best_model.pt
```
- Access the interface at http://localhost:7860
- Provides input fields and adjustable generation parameters.

---

### 3. File Translation
Translate a text file containing one English sentence per line:
```bash
python app/cli.py translate --checkpoint checkpoints/best_model.pt --input input.txt --output output.txt
```

---

### 4. REST API Server
Start the FastAPI server:
```bash
python app/api.py --checkpoint checkpoints/best_model.pt
```
- API endpoints are hosted at http://localhost:8000
- OpenAPI documentation is available at http://localhost:8000/docs

---

## Model Evaluation

To evaluate the model and check translation metrics (such as BLEU score, perplexity, and token-level accuracy), run:
```bash
python app/cli.py evaluate --checkpoint checkpoints/best_model.pt
```

---

## Training Metrics

Monitor the training metrics using TensorBoard:
```bash
tensorboard --logdir runs
```
Open http://localhost:6006 in your browser to view loss and metric curves.

---

## Troubleshooting

**Model checkpoint not found**
- Ensure that the file exists in the correct path. You can verify this by checking `checkpoints/` directory contents.

**Out of Memory (OOM) on GPU**
- Use greedy search or a smaller beam size: set `--beam-size 1` during translation/evaluation.

**Poor translation quality**
- Train the model for more epochs, or remove sample limits in the configuration to train on the full dataset.
