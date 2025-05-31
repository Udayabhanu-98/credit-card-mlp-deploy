from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from mangum import Mangum

# Initialize FastAPI app
app = FastAPI()

# Load scaler
with open("mlp_keras_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load Keras model
model = tf.keras.models.load_model("mlp_keras_model.h5")

# Pydantic model for input
class Transaction(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: Transaction):
    # Prepare and scale input
    X = np.array([data.features])
    X_scaled = scaler.transform(X)

    # Get prediction
    prob = model.predict(X_scaled)[0][0]
    label = int(prob > 0.5)

    return {
        "fraud_probability": float(prob),
        "fraud_label": label  # 1 = fraud, 0 = not fraud
    }

# For AWS Lambda (Render needs this for deployment)
lambda_handler = Mangum(app)
