# BERT Neural Machine Translation

A professional-grade English-to-French neural machine translation system using BERT encoder-decoder architecture, trained on the OPUS-100 parallel corpus.

## Features

- **State-of-the-art Architecture**: BERT encoder with transformer decoder
- **High-Quality Dataset**: Trained on OPUS-100 English-French parallel corpus
- **Multiple Interfaces**: CLI, REST API, and web UI
- **Production-Ready**: Comprehensive testing, logging, and checkpointing
- **Flexible Generation**: Supports beam search and greedy decoding
- **Mixed Precision Training**: Faster training with automatic mixed precision (AMP)
- **Evaluation Metrics**: BLEU score, perplexity, and accuracy tracking

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training](#training)
- [Translation](#translation)
- [API Usage](#api-usage)
- [Web Interface](#web-interface)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended for training)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd BERT
```

2. **Create a virtual environment**:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Quick Start

### Training a Model

Train on a small subset for testing:

```bash
python app/cli.py train --config config/training_config.yaml --num-train-samples 1000 --num-val-samples 200
```

### Translating Text

**Interactive mode**:
```bash
python app/cli.py translate --checkpoint checkpoints/best_model.pt --interactive
```

**Translate a file**:
```bash
python app/cli.py translate --checkpoint checkpoints/best_model.pt --input input.txt --output output.txt
```

### Web Interface

Launch the Gradio web interface:
```bash
python app/web_interface.py --checkpoint checkpoints/best_model.pt
```

Then open your browser to `http://localhost:7860`

## Project Structure

```
d:\BERT\
├── config/                      # Configuration files
│   ├── model_config.yaml       # Model architecture config
│   └── training_config.yaml    # Training hyperparameters
├── src/                        # Source code
│   ├── model/                  # Model components
│   │   ├── bert_translator.py  # Main translation model
│   │   └── attention.py        # Attention mechanisms
│   ├── data/                   # Data processing
│   │   ├── dataset.py          # Dataset classes
│   │   └── preprocessing.py    # Text preprocessing
│   ├── training/               # Training utilities
│   │   ├── trainer.py          # Training loop
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/                  # Utilities
│       └── tokenizer.py        # Tokenizer utilities
├── app/                        # Applications
│   ├── cli.py                  # Command-line interface
│   ├── api.py                  # REST API
│   └── web_interface.py        # Gradio web UI
├── tests/                      # Unit tests
│   ├── test_model.py           # Model tests
│   └── test_preprocessing.py   # Data tests
├── data/                       # Data directory
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── sample/                 # Sample data
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs
├── runs/                       # TensorBoard logs
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Training

### Full Training

Train on the complete OPUS-100 dataset:

```bash
python app/cli.py train --config config/training_config.yaml
```

### Training Configuration

Edit [config/training_config.yaml](file:///d:/BERT/config/training_config.yaml) to customize:

- **Batch size**: Adjust based on GPU memory
- **Learning rate**: Default is 5e-5
- **Number of epochs**: Default is 10
- **Gradient accumulation**: For larger effective batch sizes
- **Mixed precision**: Enable/disable AMP

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir runs
```

Open `http://localhost:6006` in your browser.

### Resume Training

Resume from a checkpoint:

```bash
# Modify training_config.yaml:
# resume_from_checkpoint: "checkpoints/checkpoint-step-10000.pt"

python app/cli.py train --config config/training_config.yaml
```

## Translation

### Command-Line Interface

**Interactive translation**:
```bash
python app/cli.py translate --checkpoint checkpoints/best_model.pt --interactive
```

**Batch translation**:
```bash
python app/cli.py translate \
    --checkpoint checkpoints/best_model.pt \
    --input input.txt \
    --output output.txt
```

### Python API

```python
import torch
from src.model import BERTTranslator
from src.utils import get_tokenizers

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_tokenizer, target_tokenizer = get_tokenizers()

model = BERTTranslator.from_config_file("config/model_config.yaml")
checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Translate
text = "Hello, how are you?"
inputs = source_tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    generated = model.generate(inputs["input_ids"], num_beams=5)

translation = target_tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"Translation: {translation}")
```

## API Usage

### Starting the API Server

```bash
python app/api.py --checkpoint checkpoints/best_model.pt --port 8000
```

### API Endpoints

**Translate single text**:
```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, how are you?", "num_beams": 5}'
```

**Batch translation**:
```bash
curl -X POST "http://localhost:8000/batch_translate" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Hello", "Goodbye"], "num_beams": 5}'
```

**Model information**:
```bash
curl "http://localhost:8000/model_info"
```

**Health check**:
```bash
curl "http://localhost:8000/health"
```

### API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Web Interface

### Launch Web UI

```bash
python app/web_interface.py --checkpoint checkpoints/best_model.pt
```

### Features

- **Interactive Translation**: Real-time translation with adjustable parameters
- **Example Sentences**: Pre-loaded examples to try
- **Advanced Options**: Control beam size, max length, and temperature
- **Model Information**: View model architecture and statistics
- **Modern UI**: Clean, responsive interface built with Gradio

### Public Sharing

Create a public link (valid for 72 hours):

```bash
python app/web_interface.py --checkpoint checkpoints/best_model.pt --share
```

## Model Architecture

### Overview

The model uses an encoder-decoder architecture:

1. **Encoder**: Pre-trained BERT (`bert-base-uncased`)
   - 12 transformer layers
   - 768 hidden dimensions
   - 12 attention heads

2. **Decoder**: Custom transformer decoder
   - 6 transformer layers
   - 768 hidden dimensions
   - 12 attention heads
   - Cross-attention to encoder outputs

3. **Generation**: Beam search with configurable beam size

### Key Components

- **Multi-Head Attention**: Scaled dot-product attention with multiple heads
- **Positional Encoding**: Sinusoidal position embeddings
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Improves gradient flow

### Model Size

- **Parameters**: ~110M trainable parameters
- **Vocabulary**: 
  - Source (English): 30,522 tokens (BERT vocab)
  - Target (French): ~32,000 tokens

## Configuration

### Model Configuration

Edit [config/model_config.yaml](file:///d:/BERT/config/model_config.yaml):

```yaml
model:
  encoder:
    hidden_size: 768
    num_attention_heads: 12
    num_hidden_layers: 12
  
  decoder:
    num_decoder_layers: 6
    hidden_size: 768
    num_attention_heads: 12
  
  generation:
    max_length: 128
    beam_size: 5
```

### Training Configuration

Edit [config/training_config.yaml](file:///d:/BERT/config/training_config.yaml):

```yaml
training:
  batch_size: 32
  learning_rate: 5.0e-5
  num_epochs: 10
  mixed_precision: true
  device: "cuda"
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Tests

```bash
# Test model components
pytest tests/test_model.py -v

# Test data preprocessing
pytest tests/test_preprocessing.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report at `htmlcov/index.html`

## Evaluation

### Evaluate Model

```bash
python app/cli.py evaluate --checkpoint checkpoints/best_model.pt --config config/training_config.yaml
```

### Metrics

- **BLEU Score**: Standard MT evaluation metric
- **Perplexity**: Model confidence measure
- **Accuracy**: Token-level prediction accuracy

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in training config
- Enable `gradient_accumulation_steps`
- Use smaller `max_source_length` and `max_target_length`

### Slow Training

- Enable `mixed_precision: true`
- Increase `num_workers` for data loading
- Use larger `batch_size` if GPU memory allows

### Poor Translation Quality

- Train for more epochs
- Use larger model (increase decoder layers)
- Increase training data size
- Adjust beam size during generation

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bert_translation,
  title={BERT Neural Machine Translation},
  author={Annasimon Nader},
  year={2025},
  url={https://github.com/yourusername/bert-translation}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- **OPUS-100**: Parallel corpus dataset
- **Hugging Face**: Transformers library and pre-trained models
- **PyTorch**: Deep learning framework
- **Gradio**: Web interface framework

---

**Built using BERT, PyTorch, and Transformers**
