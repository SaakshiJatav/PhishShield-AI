from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the existing trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="PhishShield AI API",
    description="Phishing email detection using NLP and Machine Learning.",
    version="1.0.0"
)


class EmailRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)


@app.get("/")
def home():
    return {
        "message": "PhishShield AI API is running",
        "status": "ok"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_email(request: EmailRequest):
    email_text = request.text.strip()

    if not email_text:
        raise HTTPException(
            status_code=400,
            detail="Email text cannot be empty."
        )

    prediction = int(model.predict([email_text])[0])

    probabilities = model.predict_proba([email_text])[0]
    classes = list(model.named_steps["classifier"].classes_)

    phishing_probability = float(
        probabilities[classes.index(1)]
    )

    return {
        "prediction": "Phishing" if prediction == 1 else "Safe",
        "label": prediction,
        "phishing_probability": round(phishing_probability, 4),
        "model": "TF-IDF + Logistic Regression",
        "warning": "Model estimate only. Not a security guarantee."
    }