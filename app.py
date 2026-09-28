"""FastAPI application for Customer Churn Intelligence."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


class CustomerProfile(BaseModel):
    """Validated fields expected by the saved model pipeline."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    senior_citizen: int = Field(ge=0, le=1)
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float = Field(ge=0, le=100000)


def to_model_frame(profile: CustomerProfile) -> pd.DataFrame:
    values = profile.model_dump()
    return pd.DataFrame(
        [
            {
                "gender": values["gender"],
                "SeniorCitizen": values["senior_citizen"],
                "Partner": values["partner"],
                "Dependents": values["dependents"],
                "tenure": values["tenure"],
                "PhoneService": values["phone_service"],
                "MultipleLines": values["multiple_lines"],
                "InternetService": values["internet_service"],
                "OnlineSecurity": values["online_security"],
                "OnlineBackup": values["online_backup"],
                "DeviceProtection": values["device_protection"],
                "TechSupport": values["tech_support"],
                "StreamingTV": values["streaming_tv"],
                "StreamingMovies": values["streaming_movies"],
                "Contract": values["contract"],
                "PaperlessBilling": values["paperless_billing"],
                "PaymentMethod": values["payment_method"],
                "MonthlyCharges": values["monthly_charges"],
                "TotalCharges": values["total_charges"],
            }
        ]
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Model artifacts are missing. Run `python -m src.train` first.")
    application.state.model = joblib.load(MODEL_PATH)
    application.state.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    yield


app = FastAPI(title="Customer Churn Intelligence", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health(request: Request):
    loaded = hasattr(request.app.state, "model")
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded}


@app.post("/predict")
async def predict(profile: CustomerProfile, request: Request):
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")

    try:
        model = request.app.state.model
        metadata = request.app.state.metadata
        frame = to_model_frame(profile)
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, 1])
    except Exception as exc:
        # Keep internal implementation details out of the API response.
        raise HTTPException(status_code=500, detail="Prediction could not be completed.") from exc

    if probability < 0.35:
        risk_level = "Low"
    elif probability < 0.65:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "model": metadata["selected_model"],
        "feature_signals": metadata.get("feature_importance", [])[:5],
    }
