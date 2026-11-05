# 🌡️ Smart Classroom AC Setpoint Prediction API

A production-ready **FastAPI** service for predicting smart classroom AC setpoint temperature (°C) using a trained **Linear Regression** model (`classroom_ac_model.pkl`).

Configured for **Render** cloud deployment with automatic preprocessing, interactive Swagger documentation, and health monitoring.

---

## 📌 Features

- **Automated Preprocessing**:
  - Automatically parses `timestamp` (`YYYY-MM-DD HH:MM:SS` or ISO format) to extract `hour` and `minute`.
  - Supports explicit `hour` and `minute` inputs.
  - Encodes categorical `day_period` (`Morning` → `0`, `Afternoon` → `1`, `Evening` → `2`) with case-insensitivity.
  - Auto-infers `day_period` from `hour` if omitted.
  - Aligns feature vectors into the exact order trained in the notebook:
    `[indoor_temp_c, indoor_humidity_pct, outdoor_temp_c, outdoor_humidity_pct, day_period, hour, minute]`
- **Model Output**:
  - Returns the exact continuous prediction (`predicted_ac_setpoint_c`).
  - Returns practical AC remote recommended setpoint rounded to nearest 0.5°C (`recommended_setpoint_c`).
  - Returns the full preprocessed feature dictionary.
- **Endpoints**:
  - `GET /` — API metadata and endpoint overview.
  - `GET /health` — Health check endpoint for Render monitoring.
  - `GET /model-info` — Model type, feature names, coefficients, and intercept.
  - `POST /predict` — Single prediction with automated preprocessing and validation.
  - `POST /predict/batch` — Batch predictions for multiple sensor readings.
- **Render Ready**: Includes `Procfile`, `render.yaml`, `requirements.txt`, and dynamic `$PORT` binding.

---

## 📂 Project Structure

```text
ML_Smart_AC_Setpoint/
├── main.py                      # FastAPI application & preprocessing logic
├── classroom_ac_model.pkl       # Trained Linear Regression model
├── Dataset.csv                  # Reference training dataset
├── requirements.txt             # Pinned production dependencies
├── Procfile                     # Web process definition for Render
├── render.yaml                  # Render Blueprint configuration
├── test_api.py                  # Unit and integration test suite
├── .gitignore                   # Ignored files
└── README.md                    # Project documentation
```

---

## ⚙️ Model & Preprocessing Summary

The model was trained on classroom environmental data:

| Feature Name | Description | Preprocessing |
| :--- | :--- | :--- |
| `indoor_temp_c` | Classroom indoor temperature (°C) | Direct float |
| `indoor_humidity_pct` | Classroom indoor relative humidity (%) | Direct float |
| `outdoor_temp_c` | Ambient outdoor temperature (°C) | Direct float |
| `outdoor_humidity_pct` | Ambient outdoor relative humidity (%) | Direct float |
| `day_period` | Time of day period | Mapped: `Morning` → 0, `Afternoon` → 1, `Evening` → 2 |
| `hour` | Hour of the reading (0 - 23) | Extracted from `timestamp` or direct integer |
| `minute` | Minute of the reading (0 - 59) | Extracted from `timestamp` or direct integer |

---

## 🚀 Local Setup & Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Test Suite
```bash
python test_api.py
```

### 3. Start the Server Locally
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🌐 API Usage & Examples

### Single Prediction (`POST /predict`)

#### Example Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "indoor_temp_c": 26.45,
       "indoor_humidity_pct": 64.51,
       "outdoor_temp_c": 26.68,
       "outdoor_humidity_pct": 74.79,
       "day_period": "Morning",
       "timestamp": "2026-11-02 08:00:00"
     }'
```

#### Example Response:
```json
{
  "status": "success",
  "predicted_ac_setpoint_c": 27.19,
  "recommended_setpoint_c": 27.0,
  "unit": "°C",
  "preprocessed_features": {
    "indoor_temp_c": 26.45,
    "indoor_humidity_pct": 64.51,
    "outdoor_temp_c": 26.68,
    "outdoor_humidity_pct": 74.79,
    "day_period": 0,
    "hour": 8,
    "minute": 0
  }
}
```

---

## ☁️ How to Deploy on Render

### Method 1: Using GitHub Web UI (Recommended)

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Add smart AC setpoint prediction API"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Create a New Web Service on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com) and log in.
   - Click **New +** > **Web Service**.
   - Connect your GitHub repository.

3. **Configure Settings**:
   - **Name**: `smart-ac-setpoint-api`
   - **Region**: Choose closest to you (e.g., Singapore, Frankfurt, Oregon)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: Free tier

4. **Environment Variables (Optional)**:
   - Add `PYTHON_VERSION` = `3.11.9` under **Environment Variables**.

5. **Deploy**:
   - Click **Create Web Service**.
   - Render will build and deploy your API. Once deployed, you will get a live URL such as:
     `https://smart-ac-setpoint-api.onrender.com`
   - Test it by visiting: `https://smart-ac-setpoint-api.onrender.com/docs`

---

### Method 2: Using Render Blueprint (`render.yaml`)

Because `render.yaml` is included in this repository:
1. In Render Dashboard, click **New +** > **Blueprint**.
2. Select your repository.
3. Render will read `render.yaml` and configure the build command, start command, health check path, and python version automatically!
4. Click **Apply**.
