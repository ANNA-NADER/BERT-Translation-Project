"""
REST API for BERT translation using FastAPI
"""

import sys
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import BERTTranslator
from src.utils import get_tokenizers


# Request/Response models
class TranslationRequest(BaseModel):
    text: str
    max_length: int | None = 128
    num_beams: int | None = 5


class BatchTranslationRequest(BaseModel):
    texts: list[str]
    max_length: int | None = 128
    num_beams: int | None = 5


class TranslationResponse(BaseModel):
    source: str
    translation: str


class BatchTranslationResponse(BaseModel):
    translations: list[TranslationResponse]


class ModelInfo(BaseModel):
    model_name: str
    source_language: str
    target_language: str
    num_parameters: int
    device: str


# Global variables
app = FastAPI(
    title="BERT Translation API",
    description="Neural machine translation API using BERT encoder-decoder",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and tokenizers (loaded on startup)
model = None
source_tokenizer = None
target_tokenizer = None
device = None
model_info = None
fallback_pipeline = None


def load_model(checkpoint_path: str, config_path: str):
    """Load model and tokenizers."""
    global model, source_tokenizer, target_tokenizer, device, model_info

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizers
    print("Loading tokenizers...")
    source_tokenizer, target_tokenizer = get_tokenizers()

    # Load model config
    model_config_path = Path(config_path).parent / "model_config.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    model_config["model"]["vocab"]["target_vocab_size"] = len(target_tokenizer)

    # Create and load model
    print("Loading model...")
    model = BERTTranslator(model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Model info
    model_info = ModelInfo(
        model_name="BERT Translation (English-French)",
        source_language="English",
        target_language="French",
        num_parameters=model.get_num_parameters(),
        device=str(device),
    )

    print("Model loaded successfully!")


def load_fallback_pipeline():
    """Load pre-trained Helsinki-NLP translation model as a fallback."""
    global fallback_pipeline, model_info, device
    print("Loading fallback Helsinki-NLP translation pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fallback_pipeline = pipeline(
        "translation_en_to_fr",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=0 if device.type == "cuda" else -1,
    )

    model_info = ModelInfo(
        model_name="Helsinki-NLP/opus-mt-en-fr (Fallback Pipeline)",
        source_language="English",
        target_language="French",
        num_parameters=sum(
            p.numel() for p in fallback_pipeline.model.parameters() if p.requires_grad
        ),
        device=str(device),
    )
    print("Fallback pipeline loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    checkpoint_path = "checkpoints/best_model.pt"
    config_path = "config/training_config.yaml"

    if Path(checkpoint_path).exists():
        load_model(checkpoint_path, config_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        load_fallback_pipeline()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BERT Translation API",
        "version": "1.0.0",
        "endpoints": {
            "translate": "/translate",
            "batch_translate": "/batch_translate",
            "model_info": "/model_info",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "fallback_loaded": fallback_pipeline is not None,
    }


@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model is None and fallback_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return model_info


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate a single text from English to French.
    """
    if model is None and fallback_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if fallback_pipeline is not None:
            # Use fallback pre-trained pipeline
            result = fallback_pipeline(
                request.text, max_length=request.max_length, num_beams=request.num_beams
            )
            translation = result[0]["translation_text"]
        else:
            # Tokenize
            inputs = source_tokenizer(
                request.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=request.max_length,
            )

            input_ids = inputs["input_ids"].to(device)

            # Generate translation
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_length=request.max_length,
                    num_beams=request.num_beams,
                )

            # Decode
            translation = target_tokenizer.decode(generated[0], skip_special_tokens=True)

        return TranslationResponse(source=request.text, translation=translation)

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.post("/batch_translate", response_model=BatchTranslationResponse)
async def batch_translate(request: BatchTranslationRequest):
    """
    Translate multiple texts from English to French.
    """
    if model is None and fallback_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        translations = []

        if fallback_pipeline is not None:
            # Use fallback pre-trained pipeline
            results = fallback_pipeline(
                request.texts,
                max_length=request.max_length,
                num_beams=request.num_beams,
            )
            for text, res in zip(request.texts, results, strict=False):
                translations.append(
                    TranslationResponse(source=text, translation=res["translation_text"])
                )
        else:
            for text in request.texts:
                # Tokenize
                inputs = source_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=request.max_length,
                )

                input_ids = inputs["input_ids"].to(device)

                # Generate translation
                with torch.no_grad():
                    generated = model.generate(
                        input_ids,
                        max_length=request.max_length,
                        num_beams=request.num_beams,
                    )

                # Decode
                translation = target_tokenizer.decode(generated[0], skip_special_tokens=True)

                translations.append(TranslationResponse(source=text, translation=translation))

        return BatchTranslationResponse(translations=translations)

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


def run_server(
    checkpoint_path: str = "checkpoints/best_model.pt",
    config_path: str = "config/training_config.yaml",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    Run the API server.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        host: Host to bind to
        port: Port to bind to
    """
    # Load model before starting server
    if Path(checkpoint_path).exists():
        load_model(checkpoint_path, config_path)

    # Run server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BERT Translation API Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    run_server(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        host=args.host,
        port=args.port,
    )
