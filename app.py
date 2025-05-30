from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from mangum import Mangum
import os

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mlp_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, **kwargs):
    # Extract features from form submission
    features = [float(kwargs[f"f{i}"]) for i in range(30)]  # assuming 30 features
    input_data = np.array([features], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = 1 if prediction >= 0.5 else 0

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": f"{prediction:.2f}",
        "label": "Fraud" if label == 1 else "Not Fraud"
    })

lambda_handler = Mangum(app)
