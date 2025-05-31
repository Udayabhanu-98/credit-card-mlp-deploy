from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from mangum import Mangum

app = FastAPI()

# Load the scaler
scaler = joblib.load("mlp_keras_scaler.pkl")

# Load the Keras model
model = tf.keras.models.load_model("mlp_keras_model.h5")

class Transaction(BaseModel):
    features: list

@app.post("/predict")
def predict(data: Transaction):
    input_data = np.array([data.features], dtype=np.float32)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    prediction_class = int(prediction > 0.3)  # lower threshold due to class imbalance
    return {
        "fraud_probability": float(prediction),
        "fraud_label": prediction_class
    }

lambda_handler = Mangum(app)
