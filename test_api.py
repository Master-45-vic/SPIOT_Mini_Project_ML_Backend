from main import (
    app,
    root,
    health,
    model_info,
    predict,
    predict_batch,
    PredictRequest,
    BatchPredictRequest,
    preprocess_single_input
)
from pydantic import ValidationError
from fastapi import HTTPException

def test_preprocessing():
    # 1. From timestamp
    p1 = preprocess_single_input(
        indoor_temp_c=26.45,
        indoor_humidity_pct=64.51,
        outdoor_temp_c=26.68,
        outdoor_humidity_pct=74.79,
        timestamp="2026-11-02 08:15:00",
        day_period="Morning"
    )
    assert p1["feature_dict"]["hour"] == 8
    assert p1["feature_dict"]["minute"] == 15
    assert p1["feature_dict"]["day_period"] == 0
    assert p1["feature_vector"] == [26.45, 64.51, 26.68, 74.79, 0, 8, 15]

    # 2. From explicit hour/minute and lower-case day_period
    p2 = preprocess_single_input(
        indoor_temp_c=22.0,
        indoor_humidity_pct=70.0,
        outdoor_temp_c=28.0,
        outdoor_humidity_pct=65.0,
        day_period="afternoon",
        hour=14,
        minute=30
    )
    assert p2["feature_dict"]["day_period"] == 1
    assert p2["feature_dict"]["hour"] == 14
    assert p2["feature_dict"]["minute"] == 30

    # 3. Inferred day_period
    p3 = preprocess_single_input(
        indoor_temp_c=22.0,
        indoor_humidity_pct=70.0,
        outdoor_temp_c=28.0,
        outdoor_humidity_pct=65.0,
        hour=19,
        minute=0
    )
    assert p3["feature_dict"]["day_period"] == 2
    print("✓ Preprocessing tests passed!")

def test_endpoints():
    r_root = root()
    assert r_root["status"] == "online"
    assert r_root["model_type"] == "LinearRegression"
    print("✓ Root endpoint passed!")

    r_health = health()
    assert r_health["status"] == "healthy"
    assert r_health["model_loaded"] is True
    print("✓ Health endpoint passed!")

    r_info = model_info()
    assert len(r_info["feature_names"]) == 7
    assert "indoor_temp_c" in r_info["coefficients"]
    print("✓ Model info endpoint passed!")

    # Test single prediction with row 0
    req = PredictRequest(
        indoor_temp_c=26.45,
        indoor_humidity_pct=64.51,
        outdoor_temp_c=26.68,
        outdoor_humidity_pct=74.79,
        day_period="Morning",
        timestamp="2026-11-02 08:00:00"
    )
    res = predict(req)
    assert res.status == "success"
    assert 26.5 <= res.predicted_ac_setpoint_c <= 27.5
    assert res.recommended_setpoint_c == 27.0
    print(f"✓ Predict endpoint passed! Predicted: {res.predicted_ac_setpoint_c}°C, Recommended: {res.recommended_setpoint_c}°C")

    # Test batch prediction
    batch_req = BatchPredictRequest(
        inputs=[
            PredictRequest(
                indoor_temp_c=26.45,
                indoor_humidity_pct=64.51,
                outdoor_temp_c=26.68,
                outdoor_humidity_pct=74.79,
                timestamp="2026-11-02 08:00:00"
            ),
            PredictRequest(
                indoor_temp_c=23.19,
                indoor_humidity_pct=69.35,
                outdoor_temp_c=27.73,
                outdoor_humidity_pct=68.35,
                day_period="Morning",
                hour=9,
                minute=10
            )
        ]
    )
    batch_res = predict_batch(batch_req)
    assert batch_res.status == "success"
    assert batch_res.count == 2
    assert len(batch_res.predictions) == 2
    print(f"✓ Batch predict endpoint passed! Count: {batch_res.count}")

def test_validation():
    # 1. Invalid humidity
    try:
        PredictRequest(
            indoor_temp_c=25.0,
            indoor_humidity_pct=150.0,
            outdoor_temp_c=30.0,
            outdoor_humidity_pct=50.0
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # 2. Invalid day_period
    try:
        preprocess_single_input(
            indoor_temp_c=25.0,
            indoor_humidity_pct=50.0,
            outdoor_temp_c=30.0,
            outdoor_humidity_pct=50.0,
            day_period="NightTime"
        )
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 422

    print("✓ Validation tests passed!")

if __name__ == "__main__":
    test_preprocessing()
    test_endpoints()
    test_validation()
    print("\nALL UNIT & INTEGRATION TESTS PASSED! 🚀")
