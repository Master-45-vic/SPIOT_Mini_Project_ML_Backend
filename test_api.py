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
        indoor_temp_c=26.44,
        indoor_humidity_pct=64.51,
        outdoor_temp_c=26.68,
        outdoor_humidity_pct=74.79,
        timestamp="2026-10-15 08:15:00",
        day_period="Morning"
    )
    assert p1["feature_dict"]["hour"] == 8
    assert p1["feature_dict"]["minute"] == 15
    assert p1["feature_dict"]["day_period"] == 0
    assert p1["feature_vector"] == [26.44, 64.51, 26.68, 74.79, 0, 8, 15]

    # 2. From time string (as used in new notebook Colab form)
    p2_time = preprocess_single_input(
        indoor_temp_c=24.0,
        indoor_humidity_pct=54.0,
        outdoor_temp_c=26.6,
        outdoor_humidity_pct=88.0,
        day_period="Morning",
        time="08:47"
    )
    assert p2_time["feature_dict"]["hour"] == 8
    assert p2_time["feature_dict"]["minute"] == 47
    assert p2_time["feature_dict"]["day_period"] == 0

    # 3. From explicit hour/minute and lower-case day_period
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

    # 4. Inferred day_period
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
    assert "classroom_ac_model_new.pkl" in r_root["model_file"]
    print("✓ Root endpoint passed! (Model: classroom_ac_model_new.pkl)")

    r_health = health()
    assert r_health["status"] == "healthy"
    assert r_health["model_loaded"] is True
    print("✓ Health endpoint passed!")

    r_info = model_info()
    assert len(r_info["feature_names"]) == 7
    assert "indoor_temp_c" in r_info["coefficients"]
    # Check new intercept: ~41.83
    assert 41.0 <= r_info["intercept"] <= 42.5
    print(f"✓ Model info endpoint passed! Intercept: {r_info['intercept']:.2f}")

    # Test single prediction with new notebook example (indoor=24, humidity=54, outdoor=26.6, out_hum=88, 08:47)
    req = PredictRequest(
        indoor_temp_c=24.0,
        indoor_humidity_pct=54.0,
        outdoor_temp_c=26.6,
        outdoor_humidity_pct=88.0,
        day_period="Morning",
        time="08:47"
    )
    res = predict(req)
    assert res.status == "success"
    # Expected prediction for this sample is ~24.58°C
    assert 24.0 <= res.predicted_ac_setpoint_c <= 25.0
    assert res.recommended_setpoint_c == 24.5
    print(f"✓ Predict endpoint passed! Predicted: {res.predicted_ac_setpoint_c}°C, Recommended: {res.recommended_setpoint_c}°C")

    # Test single prediction with Dataset2 row 0
    req2 = PredictRequest(
        indoor_temp_c=26.44,
        indoor_humidity_pct=64.51,
        outdoor_temp_c=26.68,
        outdoor_humidity_pct=74.79,
        day_period="Morning",
        timestamp="2026-10-15 08:00:00"
    )
    res2 = predict(req2)
    assert res2.status == "success"
    assert 24.5 <= res2.predicted_ac_setpoint_c <= 25.5
    print(f"✓ Predict Dataset2 row 0 passed! Predicted: {res2.predicted_ac_setpoint_c}°C, Recommended: {res2.recommended_setpoint_c}°C")

    # Test batch prediction
    batch_req = BatchPredictRequest(
        inputs=[req, req2]
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
    print("\nALL NEW MODEL UNIT & INTEGRATION TESTS PASSED! 🚀")
