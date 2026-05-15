import numpy as np
import pytest

from app.drift import DriftDetector

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is False
    assert result["n_drifted_features"] == 0


def test_drift_on_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=8.0, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is True
    assert result["n_drifted_features"] == 4
    for feat in FEATURE_NAMES:
        assert result["per_feature"][feat]["p_value"] < 0.05


def test_detector_validates_dimensions():
    ref = np.ones((100, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    with pytest.raises(ValueError):
        detector.detect(np.ones((10, 3)), alpha=0.05)


def test_detector_raises_on_1d_reference():
    with pytest.raises(ValueError):
        DriftDetector(np.ones(100), FEATURE_NAMES)


def test_check_drift_endpoint():
    """Інтеграційний тест ендпоінту /check-drift через TestClient."""
    import pytest
    from fastapi.testclient import TestClient

    from app.main import MODEL_PATH, REFERENCE_PATH, app
    from ml.train import train_and_save

    if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
        train_and_save(MODEL_PATH, REFERENCE_PATH)

    with TestClient(app) as client:
        payload = {
            "samples": [
                [5.1, 3.5, 1.4, 0.2],
                [4.9, 3.0, 1.4, 0.2],
                [4.7, 3.2, 1.3, 0.2],
                [5.4, 3.9, 1.7, 0.4],
                [5.0, 3.6, 1.4, 0.2],
                [5.5, 2.5, 4.0, 1.3],
                [6.1, 2.9, 4.7, 1.4],
                [6.0, 3.0, 4.8, 1.8],
                [6.3, 2.5, 5.0, 1.9],
                [6.5, 3.0, 5.2, 2.0],
            ],
            "alpha": 0.05,
        }
        response = client.post("/check-drift", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "drift_detected" in body
        assert "per_feature" in body
        assert "n_samples" in body
        assert body["n_samples"] == 10


def test_check_drift_detects_anomaly():
    """Перевіряє, що явний drift (аномальні значення) детектується."""
    from fastapi.testclient import TestClient

    from app.main import MODEL_PATH, REFERENCE_PATH, app
    from ml.train import train_and_save

    if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
        train_and_save(MODEL_PATH, REFERENCE_PATH)

    with TestClient(app) as client:
        payload = {
            "samples": [
                [9.0, 8.0, 8.0, 5.0],
                [9.5, 7.5, 8.5, 5.5],
                [8.5, 8.5, 7.5, 4.5],
                [9.2, 8.2, 8.2, 5.2],
                [9.8, 7.8, 8.8, 5.8],
                [8.8, 8.8, 7.8, 4.8],
                [9.4, 8.4, 8.4, 5.4],
                [9.6, 7.6, 8.6, 5.6],
                [8.6, 8.6, 7.6, 4.6],
                [9.1, 8.1, 8.1, 5.1],
            ],
            "alpha": 0.05,
        }
        response = client.post("/check-drift", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["drift_detected"] is True
        assert body["n_drifted_features"] == 4
