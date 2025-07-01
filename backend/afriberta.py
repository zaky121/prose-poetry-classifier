from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import torch
import unicodedata
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import time
from typing import Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AfriBERTa model
model_path = r"C:\Users\zakim\OneDrive\Desktop\ALL MODELS\LLMs models\poetry_prose_afriberta_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Request/Response models
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    classification: str
    processed_text: str
    is_poetry: bool
    confidence: float
    score: float
    model: str
    inference_time: float
    error: Optional[str] = None

# Text preprocessing
def preprocess_text(text: str) -> str:
    """Clean and normalize Somali text for AfriBERTa"""
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Basic cleaning
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    
    return text

# Prediction function
def predict_afriberta(text: str) -> dict:
    """Classify text as poetry or prose using AfriBERTa"""
    try:
        start_time = time.time()
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize and predict
        inputs = tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=-1)
            
        inference_time = time.time() - start_time
        
        return {
            "classification": "Tix" if predicted_class.item() == 1 else "Tiraab",
            "processed_text": processed_text,
            "is_poetry": predicted_class.item() == 1,
            "confidence": round(confidence.item() * 100, 2),
            "score": confidence.item(),
            "model": "AfriBERTa",
            "inference_time": inference_time
        }
    except Exception as e:
        return {
            "error": str(e),
            "classification": "Error",
            "processed_text": text,
            "is_poetry": False,
            "confidence": 0.0,
            "score": 0.0,
            "model": "AfriBERTa",
            "inference_time": 0.0
        }

# API Endpoints
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "model": "AfriBERTa",
        "message": "Somali Poetry/Prose Classification API - AfriBERTa",
        "version": "1.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10,000 characters)")
    
    result = predict_afriberta(request.text)
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)