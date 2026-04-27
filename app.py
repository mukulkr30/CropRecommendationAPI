from fastapi import FastAPI
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os

app = FastAPI()

# CORS (restrict later in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://digi-agro.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ FIX: absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "Crop_recommendation_model.pkl"))
le = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: dict):
    features = [
        data["nitrogen"],
        data["phosphorus"],
        data["potassium"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]

    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]
    crop = le.inverse_transform([pred])[0]

    return {"crop": crop}