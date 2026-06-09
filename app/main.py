from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Weight Status Prediction")
templates = Jinja2Templates(directory="app/templates")

MODEL_PATH = Path("Model/health_status_model.pkl")
model = joblib.load(MODEL_PATH)


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    if height_m <= 0:
        raise ValueError("Height must be greater than 0.")
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Healthy"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def empty_form():
    return {
        "age": "",
        "gender": "Male",
        "height_cm": "",
        "weight_kg": "",
        "activity_level": "Sedentary",
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": None,
            "bmi": None,
            "bmi_category": None,
            "probabilities": None,
            "form": empty_form(),
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    activity_level: str = Form(...),
):
    bmi = calculate_bmi(height_cm, weight_kg)

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "BMI": bmi,
        "Activity_Level": activity_level,
    }])

    prediction = model.predict(input_data)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        probabilities = {
            str(label): float(prob)
            for label, prob in zip(model.classes_, proba)
        }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": prediction,
            "bmi": bmi,
            "bmi_category": bmi_category(bmi),
            "probabilities": probabilities,
            "form": {
                "age": age,
                "gender": gender,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "activity_level": activity_level,
            },
        },
    )