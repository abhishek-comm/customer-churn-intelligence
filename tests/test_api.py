from fastapi.testclient import TestClient

from app import app

SAMPLE = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 79.85,
    "total_charges": 958.2,
}


def test_health_and_prediction():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        prediction = client.post("/predict", json=SAMPLE)
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["prediction"] in {"Yes", "No"}
        assert 0 <= body["churn_probability"] <= 1
        assert body["risk_level"] in {"Low", "Medium", "High"}
        assert body["model"]


def test_invalid_input_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict", json={**SAMPLE, "tenure": -1})
        assert response.status_code == 422
