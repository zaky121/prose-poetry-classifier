import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import joblib
from tensorflow.keras.models import load_model
import unicodedata
import numpy as np
from typing import Optional
import logging
from fastapi.middleware.cors import CORSMiddleware
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Somali Poetry/Prose Classifier API (RNN)",
    description="RNN Model with TF-IDF features for classifying Somali text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer with error handling
try:
    logger.info("Loading model and vectorizer...")
    tfidf_vectorizer = joblib.load(r'C:\Users\zakim\OneDrive\Desktop\DL\RNN\uitest\tfidf_vectorizer.joblib')
    tfidf_model = load_model(r'C:\Users\zakim\OneDrive\Desktop\DL\RNN\uitest\rnn_tfidf_model.h5')
    logger.info("Model and vectorizer loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Somali alphabet
SOMALI_ALPHABET = (
    "aA" "bB" "cC" "dD" "eE" "fF" "gG" "hH" "iI" "jJ" "kK" "lL" "mM" "nN"
    "oO" "qQ" "rR" "sS" "tT" "uU" "wW" "xX" "yY"
)
ALLOWED_CHARS = SOMALI_ALPHABET + " ,.!?()-'"

def preprocessor(text):
    text = unicodedata.normalize("NFKD", text)
    text = ''.join([char for char in text if not unicodedata.combining(char)])
    text = text.lower()
    text = ''.join([char for char in text if char in ALLOWED_CHARS.lower()])
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([,.!?()-])\1+', r'\1', text)

    # Remove repeated words (3 or more)
    text = re.sub(r'\b(\w+)( \1){2,}\b', r'\1 \1', text)

    # Remove repeated phrase chunks (3+)
    text = re.sub(r'(\b.+?\b)( \1){2,}', r'\1', text)

    # Remove repeated sentence-level lines
    lines = re.split(r'(?<=[.!?])\s+|\n+', text)
    seen = set()
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            filtered_lines.append(line)
    text = ' '.join(filtered_lines)

    return text

class PredictionRequest(BaseModel):
    text: str
    min_word_count: Optional[int] = 5
    min_char_count: Optional[int] = 30

class PredictionResponse(BaseModel):
    classification: str
    processed_text: str
    is_poetry: bool
    confidence: float
    score: float
    model: str
    inference_time: float
    is_error: bool = False
    error_message: Optional[str] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify Somali text as Poetry (Tix) or Prose (Tiraab)"""
    try:
        start_time = time.time()
        processed_text = preprocessor(request.text)
        
        # Validate input length
        if len(processed_text.strip()) < request.min_char_count:
            return {
                "classification": "Invalid",
                "processed_text": processed_text,
                "is_poetry": False,
                "confidence": 0.0,
                "score": 0.0,
                "model": "RNN",
                "inference_time": 0.0,
                "is_error": True,
                "error_message": f"Text too short (min {request.min_char_count} characters required)"
            }

        if len(processed_text.strip().split()) < request.min_word_count:
            return {
                "classification": "Invalid",
                "processed_text": processed_text,
                "is_poetry": False,
                "confidence": 0.0,
                "score": 0.0,
                "model": "RNN",
                "inference_time": 0.0,
                "is_error": True,
                "error_message": f"Text too short (min {request.min_word_count} words required)"
            }

        # Transform and predict
        features = tfidf_vectorizer.transform([processed_text]).toarray()
        input_data = features.reshape((1, 1, features.shape[1]))
        prediction = tfidf_model.predict(input_data, verbose=0)[0][0]
        
        predicted_class = 1 if prediction > 0.5 else 0
        class_label = "Tix" if predicted_class == 1 else "Tiraab"
        confidence = prediction if predicted_class == 1 else 1 - prediction
        inference_time = time.time() - start_time

        return {
            "classification": class_label,
            "processed_text": processed_text,
            "is_poetry": predicted_class == 1,
            "confidence": float(confidence * 100),
            "score": float(prediction),
            "model": "RNN",
            "inference_time": inference_time
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )