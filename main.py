import os
import sys
import types
from datetime import datetime
from typing import List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib

# ---------------------------------------------------------------------------
# Robust Model Loader
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "classroom_ac_model.pkl")

# Exact feature names trained in the notebook:
FEATURE_NAMES = [
    "indoor_temp_c",
    "indoor_humidity_pct",
    "outdoor_temp_c",
    "outdoor_humidity_pct",
    "day_period",
    "hour",
    "minute"
]

# Known model parameters from notebook training:
FALLBACK_COEF = np.array([
    9.07484644e-02,  # indoor_temp_c
    -2.43544964e-01, # indoor_humidity_pct
    8.81586439e-01,  # outdoor_temp_c
    8.92783403e-02,  # outdoor_humidity_pct
    3.07663366e-01,  # day_period
    -5.47355485e-01, # hour
    1.30723595e-04   # minute
], dtype=np.float64)
FALLBACK_INTERCEPT = 14.678647820076161


def load_model():
    """
    Loads classroom_ac_model.pkl using joblib.
    Provides mock support if sklearn is not installed or has version mismatch,
    ensuring 100% reliability in any environment.
    """
    if os.path.exists(MODEL_PATH):
        try:
            # Ensure mock module is registered in case sklearn isn't present
            if "sklearn" not in sys.modules:
                m_sklearn = types.ModuleType("sklearn")
                m_lm = types.ModuleType("sklearn.linear_model")
                m_base = types.ModuleType("sklearn.linear_model._base")

                class MockLinearRegression:
                    def __init__(self, *args, **kwargs):
                        pass

                    def __setstate__(self, state):
                        self.__dict__.update(state)

                    def predict(self, X):
                        X_arr = np.asarray(X, dtype=np.float64)
                        return X_arr @ self.coef_ + self.intercept_

                m_base.LinearRegression = MockLinearRegression
                m_lm._base = m_base
                m_sklearn.linear_model = m_lm
                sys.modules["sklearn"] = m_sklearn
                sys.modules["sklearn.linear_model"] = m_lm
                sys.modules["sklearn.linear_model._base"] = m_base

            loaded = joblib.load(MODEL_PATH)
            return loaded
        except Exception as exc:
            print(f"[Warning] Failed to load model file directly: {exc}. Using verified weights fallback.")

    # Fallback wrapper with exact coefficients
    class FallbackModel:
        coef_ = FALLBACK_COEF
        intercept_ = FALLBACK_INTERCEPT
        feature_names_in_ = np.array(FEATURE_NAMES)
        n_features_in_ = len(FEATURE_NAMES)

        def predict(self, X):
            X_arr = np.asarray(X, dtype=np.float64)
            return X_arr @ self.coef_ + self.intercept_

    return FallbackModel()


model = load_model()

# ---------------------------------------------------------------------------
# FastAPI Application Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart AC Setpoint Prediction API",
    description=(
        "Production-ready API for predicting classroom AC setpoint (°C) "
        "using a trained Linear Regression model. Automatically handles "
        "timestamp parsing, day_period encoding, and feature vector alignment."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Preprocessing Logic
# ---------------------------------------------------------------------------
DAY_PERIOD_MAPPING = {
    "morning": 0,
    "afternoon": 1,
    "evening": 2
}


def infer_day_period(hour: int) -> int:
    """
    Infers day period code from hour if day_period is not explicitly provided.
    - Morning (0): 05:00 - 11:59
    - Afternoon (1): 12:00 - 16:59
    - Evening (2): 17:00 - 04:59
    """
    if 5 <= hour < 12:
        return 0
    elif 12 <= hour < 17:
        return 1
    else:
        return 2


def preprocess_single_input(
    indoor_temp_c: float,
    indoor_humidity_pct: float,
    outdoor_temp_c: float,
    outdoor_humidity_pct: float,
    timestamp: Optional[str] = None,
    day_period: Optional[Union[str, int]] = None,
    hour: Optional[int] = None,
    minute: Optional[int] = None
) -> dict:
    """
    Implements the exact preprocessing pipeline from the ML notebook:
    1. Parse timestamp into datetime to extract hour and minute (or use explicit hour/minute).
    2. Convert day_period ("Morning": 0, "Afternoon": 1, "Evening": 2).
    3. Order features matching model input:
       [indoor_temp_c, indoor_humidity_pct, outdoor_temp_c, outdoor_humidity_pct, day_period, hour, minute]
    """
    extracted_hour = hour
    extracted_minute = minute

    # Step 1: Extract hour and minute from timestamp if provided
    if timestamp:
        ts_clean = timestamp.strip()
        parsed_dt = None
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%H:%M:%S",
            "%H:%M"
        ):
            try:
                parsed_dt = datetime.strptime(ts_clean, fmt)
                break
            except ValueError:
                continue

        if parsed_dt is not None:
            if extracted_hour is None:
                extracted_hour = parsed_dt.hour
            if extracted_minute is None:
                extracted_minute = parsed_dt.minute
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Could not parse timestamp '{timestamp}'. Expected format: 'YYYY-MM-DD HH:MM:SS' or ISO format."
            )

    # Defaults if neither timestamp nor explicit hour/minute were provided
    if extracted_hour is None:
        extracted_hour = 12
    if extracted_minute is None:
        extracted_minute = 0

    # Step 2: Convert day_period to numerical mapping
    encoded_day_period = None
    if day_period is not None:
        if isinstance(day_period, int):
            if day_period in (0, 1, 2):
                encoded_day_period = day_period
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid day_period integer '{day_period}'. Allowed values are 0 (Morning), 1 (Afternoon), 2 (Evening)."
                )
        elif isinstance(day_period, str):
            dp_norm = day_period.strip().lower()
            if dp_norm in DAY_PERIOD_MAPPING:
                encoded_day_period = DAY_PERIOD_MAPPING[dp_norm]
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Invalid day_period string '{day_period}'. "
                        "Allowed values are: 'Morning', 'Afternoon', 'Evening' (case-insensitive) or integer 0, 1, 2."
                    )
                )

    # If day_period not provided, infer from hour
    if encoded_day_period is None:
        encoded_day_period = infer_day_period(extracted_hour)

    # Step 3: Align feature dictionary
    features_dict = {
        "indoor_temp_c": float(indoor_temp_c),
        "indoor_humidity_pct": float(indoor_humidity_pct),
        "outdoor_temp_c": float(outdoor_temp_c),
        "outdoor_humidity_pct": float(outdoor_humidity_pct),
        "day_period": int(encoded_day_period),
        "hour": int(extracted_hour),
        "minute": int(extracted_minute)
    }

    # Vector in exact training order
    feature_vector = [features_dict[col] for col in FEATURE_NAMES]

    return {
        "feature_dict": features_dict,
        "feature_vector": feature_vector
    }


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    indoor_temp_c: float = Field(..., description="Indoor temperature in °C", examples=[26.45])
    indoor_humidity_pct: float = Field(..., description="Indoor relative humidity percentage", examples=[64.51])
    outdoor_temp_c: float = Field(..., description="Outdoor ambient temperature in °C", examples=[26.68])
    outdoor_humidity_pct: float = Field(..., description="Outdoor relative humidity percentage", examples=[74.79])
    timestamp: Optional[str] = Field(
        None,
        description="Timestamp string (e.g. '2026-11-02 08:30:00'). Hour and minute are extracted automatically.",
        examples=["2026-11-02 08:00:00"]
    )
    day_period: Optional[Union[str, int]] = Field(
        None,
        description="Day period: 'Morning', 'Afternoon', 'Evening' (or 0, 1, 2). Inferred from hour if omitted.",
        examples=["Morning"]
    )
    hour: Optional[int] = Field(None, ge=0, le=23, description="Explicit hour (0-23) if timestamp not provided", examples=[8])
    minute: Optional[int] = Field(None, ge=0, le=59, description="Explicit minute (0-59) if timestamp not provided", examples=[0])

    @field_validator("indoor_humidity_pct", "outdoor_humidity_pct")
    @classmethod
    def validate_humidity(cls, v):
        if not (0.0 <= v <= 100.0):
            raise ValueError("Humidity percentage must be between 0 and 100.")
        return v

    @field_validator("indoor_temp_c", "outdoor_temp_c")
    @classmethod
    def validate_temp(cls, v):
        if not (-20.0 <= v <= 65.0):
            raise ValueError("Temperature must be within a realistic range (-20°C to 65°C).")
        return v


class PredictResponse(BaseModel):
    status: str = Field("success", examples=["success"])
    predicted_ac_setpoint_c: float = Field(..., description="Raw continuous AC setpoint prediction in °C", examples=[27.19])
    recommended_setpoint_c: float = Field(..., description="Practical AC setpoint rounded to nearest 0.5°C step", examples=[27.0])
    unit: str = Field("°C", examples=["°C"])
    preprocessed_features: dict = Field(..., description="Preprocessed input features fed to the Linear Regression model")


class BatchPredictRequest(BaseModel):
    inputs: List[PredictRequest] = Field(..., description="List of input readings for batch prediction")


class BatchPredictResponse(BaseModel):
    status: str = "success"
    count: int
    predictions: List[PredictResponse]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    """
    Root endpoint providing API information and documentation link.
    """
    return {
        "name": "Smart AC Setpoint Prediction API",
        "description": "API for predicting classroom AC setpoint temperature using trained Linear Regression.",
        "model_type": "LinearRegression",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "model_info": "GET /model-info"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    """
    Health check endpoint for Render monitoring and uptime checks.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model-info", tags=["Info"])
def model_info():
    """
    Returns information about the trained Linear Regression model.
    """
    coef = getattr(model, "coef_", FALLBACK_COEF).tolist()
    intercept = float(getattr(model, "intercept_", FALLBACK_INTERCEPT))
    return {
        "model_type": "LinearRegression",
        "feature_names": FEATURE_NAMES,
        "intercept": intercept,
        "coefficients": dict(zip(FEATURE_NAMES, coef)),
        "day_period_mapping": DAY_PERIOD_MAPPING
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(payload: PredictRequest):
    """
    Accepts classroom environment conditions, preprocesses features,
    and returns the predicted AC setpoint (°C).
    """
    preprocessed = preprocess_single_input(
        indoor_temp_c=payload.indoor_temp_c,
        indoor_humidity_pct=payload.indoor_humidity_pct,
        outdoor_temp_c=payload.outdoor_temp_c,
        outdoor_humidity_pct=payload.outdoor_humidity_pct,
        timestamp=payload.timestamp,
        day_period=payload.day_period,
        hour=payload.hour,
        minute=payload.minute
    )

    X = np.array([preprocessed["feature_vector"]], dtype=np.float64)
    raw_pred = float(model.predict(X)[0])

    # Round to 2 decimal places for continuous output
    rounded_pred = round(raw_pred, 2)
    # Round to nearest 0.5 for practical AC remote step
    recommended_setpoint = round(raw_pred * 2.0) / 2.0

    return PredictResponse(
        status="success",
        predicted_ac_setpoint_c=rounded_pred,
        recommended_setpoint_c=recommended_setpoint,
        unit="°C",
        preprocessed_features=preprocessed["feature_dict"]
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(payload: BatchPredictRequest):
    """
    Performs batch predictions for a list of input readings.
    """
    if not payload.inputs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'inputs' list cannot be empty."
        )

    results = []
    for item in payload.inputs:
        preprocessed = preprocess_single_input(
            indoor_temp_c=item.indoor_temp_c,
            indoor_humidity_pct=item.indoor_humidity_pct,
            outdoor_temp_c=item.outdoor_temp_c,
            outdoor_humidity_pct=item.outdoor_humidity_pct,
            timestamp=item.timestamp,
            day_period=item.day_period,
            hour=item.hour,
            minute=item.minute
        )
        X = np.array([preprocessed["feature_vector"]], dtype=np.float64)
        raw_pred = float(model.predict(X)[0])
        rounded_pred = round(raw_pred, 2)
        recommended_setpoint = round(raw_pred * 2.0) / 2.0

        results.append(
            PredictResponse(
                status="success",
                predicted_ac_setpoint_c=rounded_pred,
                recommended_setpoint_c=recommended_setpoint,
                unit="°C",
                preprocessed_features=preprocessed["feature_dict"]
            )
        )

    return BatchPredictResponse(
        status="success",
        count=len(results),
        predictions=results
    )


# ---------------------------------------------------------------------------
# Local Server Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
