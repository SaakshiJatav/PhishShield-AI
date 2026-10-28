from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_phishing_email():
    email = (
        "URGENT: Your account has been suspended. "
        "Verify your password immediately."
    )
    response = client.post("/predict", json={"text": email})

    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] == "Phishing"
    assert data["label"] == 1
    assert 0.0 <= data["phishing_probability"] <= 1.0


def test_safe_email():
    email = (
        "Hi Alex, our project meeting is tomorrow at 10 AM. "
        "Please bring your notes. Thanks, Sarah."
    )
    response = client.post("/predict", json={"text": email})

    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] == "Safe"
    assert data["label"] == 0


def test_missing_email():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_blank_email():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


def test_email_too_long():
    response = client.post(
        "/predict",
        json={"text": "A" * 50001}
    )
    assert response.status_code == 422