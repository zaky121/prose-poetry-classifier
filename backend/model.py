from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import torch
import unicodedata
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Somali Poetry/Prose Classifier API",
    description="API for classifying Somali text as poetry (Tix) or prose (Tiraab)",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading
model_path = r"C:\Users\zakim\OneDrive\Desktop\ALL MODELS\LLMs models\poetry_prose_somberta_model"
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("SoBERTa model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model") from e

# Request/Response models
class TextRequest(BaseModel):
    text: str
    model: str = "soberta"  # Default to SoBERTa

class PredictionResponse(BaseModel):
    classification: str
    processed_text: str
    is_poetry: bool
    confidence: float
    score: float
    model: str
    inference_time: float

# Constants
SOMALI_ALPHABET = (
    "aA" "bB" "cC" "dD" "eE" "fF" "gG" "hH" "iI" "jJ" "kK" "lL" "mM" "nN"
    "oO" "qQ" "rR" "sS" "tT" "uU" "wW" "xX" "yY"
)
ALLOWED_CHARS = SOMALI_ALPHABET + " ,.!?()-'"

# Text preprocessing
def preprocessor(text: str) -> str:
    """Clean and normalize Somali text"""
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
    
    # Remove repeated sentences
    lines = re.split(r'(?<=[.!?])\s+|\n+', text)
    seen = set()
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            filtered_lines.append(line)
    return ' '.join(filtered_lines)

# Prediction function
def predict(text: str, model, tokenizer, max_length=512, min_word_count=5, min_char_count=30) -> tuple:
    """Classify text as poetry or prose"""
    # Basic length checks
    if len(text.strip()) < min_char_count or len(text.strip().split()) < min_word_count:
        return ("Tiraab", 0.0)  # Default to prose for short texts
    
    # Tokenization and prediction
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = float(probabilities[predicted_class])
    
    return ("Tix" if predicted_class == 1 else "Tiraab", confidence)

# API Endpoints
@app.get("/")
async def health_check():
    return {
        "message": "Somali Poetry/Prose Classification API",
        "status": "healthy",
        "model": "SoBERTa",
        "ready": True
    }

@app.post("/predict", response_model=PredictionResponse)
async def classify_text(request: TextRequest):
    try:
        import time
        start_time = time.time()
        
        processed_text = preprocessor(request.text)
        classification, confidence = predict(processed_text, model, tokenizer)
        inference_time = time.time() - start_time
        
        return {
            "classification": classification,
            "processed_text": processed_text,
            "is_poetry": classification == "Tix",
            "confidence": confidence * 100,  # as percentage
            "score": confidence,           # raw score
            "model": "soberta",
            "inference_time": inference_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)