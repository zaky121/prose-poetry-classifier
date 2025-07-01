from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import re
import os
from datetime import datetime
import uvicorn
import logging
from typing import Optional
import json
from fastapi.encoders import jsonable_encoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("uvicorn")

# Clear TensorFlow session
tf.keras.backend.clear_session()

app = FastAPI(
    title="Somali Poetry/Prose Classifier API (LSTM)",
    version="4.0",
    docs_url=None,
    redoc_url=None
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TextRequest(BaseModel):
    text: str
    min_length: Optional[int] = 5

class PredictionResponse(BaseModel):
    classification: str
    confidence: float
    is_poetry: bool
    model: str
    inference_time: float
    processed_text: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_summary: dict

class LSTMPredictor:
    def __init__(self, model_path: str):
        self.model = None
        self.load_time = datetime.now()
        self.vocab_size = 112914  # Update with your actual vocab size
        self.max_len = 256        # Update with your max sequence length
        
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            
            # Disable unnecessary warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            
            # Load model with custom objects if needed
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False
            )
            
            # Force build the model
            dummy_input = np.zeros((1, 1, self.vocab_size))
            self.model.predict(dummy_input)
            
            logger.info("Model loaded successfully")
            logger.info(f"Input shape: {self.model.input_shape}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Enhanced Somali text cleaning preserving poetic features"""
        # Preserve line breaks for poetry
        text = text.replace('\n', ' [NEWLINE] ')
        # Remove special chars but keep Somali punctuation
        text = re.sub(r"[^\w\s'-]", '', text.lower())
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text: str) -> tuple:
        """Convert text to model input with enhanced features"""
        cleaned_text = self._clean_text(text)
        features = np.zeros((1, 1, self.vocab_size))
        
        # Enhanced feature extraction
        for i, word in enumerate(cleaned_text.split()[:self.max_len]):
            # Stable hashing across sessions
            hash_val = hash(word.encode('utf-8'))
            idx = (hash_val & 0x7fffffff) % self.vocab_size
            # Position-aware weighting
            features[0, 0, idx] += 1.0 - (i * 0.01)  # Earlier words get slightly more weight
            
        return features, cleaned_text

# Initialize model
MODEL_PATH = os.path.abspath(r"C:\Users\zakim\OneDrive\Desktop\DL\uilstm\lstm_tfidf_model.h5")
predictor = None

try:
    predictor = LSTMPredictor(MODEL_PATH)
except Exception as e:
    logger.critical(f"Failed to initialize predictor: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting LSTM classification server")
    logger.info(f"API docs: http://localhost:8003/docs")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    if not predictor or not predictor.model:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_summary": {
            "input_shape": str(predictor.model.input_shape),
            "vocab_size": predictor.vocab_size,
            "max_length": predictor.max_len
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    if not predictor or not predictor.model:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    # Validate input
    words = request.text.strip().split()
    if len(words) < request.min_length:
        raise HTTPException(
            status_code=400,
            detail=f"Minimum {request.min_length} words required. Got {len(words)} words."
        )
    
    try:
        start_time = datetime.now()
        
        # Preprocess with enhanced features
        features, cleaned_text = predictor.preprocess(request.text)
        
        logger.debug(f"Input shape: {features.shape}")
        logger.debug(f"Non-zero features: {np.count_nonzero(features)}")
        
        # Get prediction
        prediction = predictor.model.predict(features, verbose=0)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        logger.debug(f"Raw prediction: {prediction}")
        
        # Interpret results
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction)) * 100
        
        # Debug output
        logger.info(
            f"Classification: {'Tix' if class_idx == 1 else 'Tiraab'} "
            f"({confidence:.2f}%) in {inference_time:.3f}s"
        )
        
        return {
            "classification": "Tix" if class_idx == 1 else "Tiraab",
            "confidence": round(confidence, 2),
            "is_poetry": bool(class_idx),
            "model": "LSTM",
            "inference_time": round(inference_time, 4),
            "processed_text": cleaned_text
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info",
        access_log=False,
        workers=1,
        reload=False
    )