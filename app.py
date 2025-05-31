from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf  # ✅ use tf instead of tflite_runtime
from mangum import Mangum

app = FastAPI()

# Root route to verify app is live
@app.get("/")
def read_root():
    return {"message": "MLP Credit Card Fraud Detection API is Live!"}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mlp_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class Transaction(BaseModel):
    features: list

@app.post("/predict")
def predict(data: Transaction):
    input_data = np.array([data.features], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    prediction_class = int(prediction > 0.5)  # threshold can be changed if needed
    return {
        "fraud_probability": float(prediction),
        "fraud_label": prediction_class  # 1 = Fraud, 0 = Not Fraud
    }

lambda_handler = Mangum(app)
