# 🌡️ Smart Classroom AC Setpoint Prediction API (ML Backend)

[![Render](https://img.shields.io/badge/Render-Live-brightgreen?logo=render)](https://spiot-mini-project-ml-backend.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-classroom__ac__model__new.pkl-blueviolet)](https://github.com/Master-45-vic/SPIOT_Mini_Project_ML_Backend)
[![Swagger Docs](https://img.shields.io/badge/API_Docs-Swagger_UI-85EA2D?logo=swagger)](https://spiot-mini-project-ml-backend.onrender.com/docs)

Production ML backend API for predicting optimal classroom **Air Conditioner Setpoint Temperatures (°C)** using a trained **Linear Regression** model (`classroom_ac_model_new.pkl`) trained on **1,635 sensor records** (`Dataset2.csv`).

The API automatically handles IoT sensor preprocessing, timestamp and time string parsing, categorical encoding, and feature vector alignment.

---

## 🌐 Live Cloud Deployment (Render)

- **Live Base URL**: [`https://spiot-mini-project-ml-backend.onrender.com`](https://spiot-mini-project-ml-backend.onrender.com)
- **Interactive Swagger UI (Test in Browser)**: [`https://spiot-mini-project-ml-backend.onrender.com/docs`](https://spiot-mini-project-ml-backend.onrender.com/docs)
- **Alternative ReDoc UI**: [`https://spiot-mini-project-ml-backend.onrender.com/redoc`](https://spiot-mini-project-ml-backend.onrender.com/redoc)
- **Health Check Endpoint**: [`https://spiot-mini-project-ml-backend.onrender.com/health`](https://spiot-mini-project-ml-backend.onrender.com/health)

---

## 📋 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status, active model file (`classroom_ac_model_new.pkl`), and endpoint directory. |
| `GET` | `/health` | Health check for uptime monitoring (returns `model_loaded: true`). |
| `GET` | `/model-info` | Model metadata, intercept (`41.8328`), and feature slope coefficients. |
| `POST` | `/predict` | **Single Prediction**: Accepts classroom conditions, preprocesses, and predicts AC setpoint. |
| `POST` | `/predict/batch` | **Batch Prediction**: Predicts setpoint for a list of multiple sensor readings. |

---

## 📥 Input Format Specification

You can send data either using a **timestamp string**, a **time string** (`"08:47"`), OR by providing **hour and minute** directly.

### Fields

| Field Name | Type | Required? | Example | Description |
| :--- | :--- | :---: | :--- | :--- |
| `indoor_temp_c` | `float` | **Yes** | `24.0` | Classroom indoor temperature in °C (-20°C to 65°C). |
| `indoor_humidity_pct` | `float` | **Yes** | `54.0` | Classroom indoor relative humidity in % (0% to 100%). |
| `outdoor_temp_c` | `float` | **Yes** | `26.6` | Outdoor ambient temperature in °C (-20°C to 65°C). |
| `outdoor_humidity_pct` | `float` | **Yes** | `88.0` | Outdoor ambient relative humidity in % (0% to 100%). |
| `time` | `string` | *Optional* | `"08:47"` | Time string (`"HH:MM"` or `"HH:MM:SS"`). Extracts `hour` and `minute`. |
| `timestamp` | `string` | *Optional* | `"2026-10-15 08:47:00"` | Timestamp (`YYYY-MM-DD HH:MM:SS`). Extracts `hour` and `minute`. |
| `day_period` | `string` / `int` | *Optional* | `"Morning"` | `"Morning"` (0), `"Afternoon"` (1), `"Evening"` (2). Auto-inferred if omitted. |
| `hour` | `int` | *Optional* | `8` | Hour of reading (0–23). Used if `time`/`timestamp` is omitted. |
| `minute` | `int` | *Optional* | `47` | Minute of reading (0–59). Used if `time`/`timestamp` is omitted. |

---

## 📤 Output Format Specification

The response returns the exact ML prediction, a rounded AC remote setpoint, and the preprocessed features:

```json
{
  "status": "success",
  "predicted_ac_setpoint_c": 24.58,
  "recommended_setpoint_c": 24.5,
  "unit": "°C",
  "preprocessed_features": {
    "indoor_temp_c": 24.0,
    "indoor_humidity_pct": 54.0,
    "outdoor_temp_c": 26.6,
    "outdoor_humidity_pct": 88.0,
    "day_period": 0,
    "hour": 8,
    "minute": 47
  }
}
```

- **`predicted_ac_setpoint_c`**: Continuous output from the Linear Regression model (rounded to 2 decimal places).
- **`recommended_setpoint_c`**: Practical AC setpoint rounded to the nearest 0.5°C increment (compatible with commercial AC controllers).
- **`preprocessed_features`**: The exact numerical vector values passed to the model after parsing and encoding.

---

## 💻 How to Use the API

### 1. Interactive Web Interface (No Code Needed)
1. Open [`https://spiot-mini-project-ml-backend.onrender.com/docs`](https://spiot-mini-project-ml-backend.onrender.com/docs) in your browser.
2. Click on **`POST /predict`** $\rightarrow$ **Try it out**.
3. Edit the JSON body and click **Execute**.

---

### 2. cURL (Command Line / Terminal)

#### Single Prediction (with `time` string):
```bash
curl -X POST "https://spiot-mini-project-ml-backend.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "indoor_temp_c": 24.0,
       "indoor_humidity_pct": 54.0,
       "outdoor_temp_c": 26.6,
       "outdoor_humidity_pct": 88.0,
       "day_period": "Morning",
       "time": "08:47"
     }'
```

#### Single Prediction (with full `timestamp`):
```bash
curl -X POST "https://spiot-mini-project-ml-backend.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "indoor_temp_c": 26.44,
       "indoor_humidity_pct": 64.51,
       "outdoor_temp_c": 26.68,
       "outdoor_humidity_pct": 74.79,
       "day_period": "Morning",
       "timestamp": "2026-10-15 08:00:00"
     }'
```

#### Batch Prediction:
```bash
curl -X POST "https://spiot-mini-project-ml-backend.onrender.com/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "inputs": [
         {
           "indoor_temp_c": 24.0,
           "indoor_humidity_pct": 54.0,
           "outdoor_temp_c": 26.6,
           "outdoor_humidity_pct": 88.0,
           "time": "08:47",
           "day_period": "Morning"
         },
         {
           "indoor_temp_c": 26.44,
           "indoor_humidity_pct": 64.51,
           "outdoor_temp_c": 26.68,
           "outdoor_humidity_pct": 74.79,
           "timestamp": "2026-10-15 08:00:00",
           "day_period": "Morning"
         }
       ]
     }'
```

---

### 3. Python (`requests`)

```python
import requests

url = "https://spiot-mini-project-ml-backend.onrender.com/predict"

payload = {
    "indoor_temp_c": 24.0,
    "indoor_humidity_pct": 54.0,
    "outdoor_temp_c": 26.6,
    "outdoor_humidity_pct": 88.0,
    "day_period": "Morning",
    "time": "08:47"
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Predicted AC Setpoint: {data['predicted_ac_setpoint_c']} °C")
print(f"Recommended Setpoint : {data['recommended_setpoint_c']} °C")
```

---

### 4. JavaScript / TypeScript (`fetch`)

```javascript
const response = await fetch("https://spiot-mini-project-ml-backend.onrender.com/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    indoor_temp_c: 24.0,
    indoor_humidity_pct: 54.0,
    outdoor_temp_c: 26.6,
    outdoor_humidity_pct: 88.0,
    day_period: "Morning",
    time: "08:47"
  })
});

const data = await response.json();
console.log(`Recommended Setpoint: ${data.recommended_setpoint_c} °C`);
```

---

### 5. ESP32 / Arduino (C++) IoT Integration

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* serverUrl = "https://spiot-mini-project-ml-backend.onrender.com/predict";

void sendSensorData(float inTemp, float inHum, float outTemp, float outHum, int hour, int minute) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<256> doc;
    doc["indoor_temp_c"] = inTemp;
    doc["indoor_humidity_pct"] = inHum;
    doc["outdoor_temp_c"] = outTemp;
    doc["outdoor_humidity_pct"] = outHum;
    doc["hour"] = hour;
    doc["minute"] = minute;

    String requestBody;
    serializeJson(doc, requestBody);

    int httpResponseCode = http.POST(requestBody);
    if (httpResponseCode == 200) {
      String response = http.getString();
      StaticJsonDocument<512> resDoc;
      deserializeJson(resDoc, response);
      float recommended = resDoc["recommended_setpoint_c"];
      Serial.printf("Adjust AC Setpoint to: %.1f °C\n", recommended);
    }
    http.end();
  }
}
```

---

## 🧠 Model Training Details (New Model)

- **Dataset**: `Dataset2.csv` (1,635 rows)
- **Model Type**: Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Trained Model File**: `classroom_ac_model_new.pkl`
- **Training Script**: `ml_model_new_notebook_python_file.py` / `ML_Model_New_Notebbok.ipynb`
- **Trained Parameters**:
  - **Intercept ($c$)**: `41.8328`
  - **Coefficients**:
    - `indoor_temp_c`: `+0.011553`
    - `indoor_humidity_pct`: `-0.018828`
    - `outdoor_temp_c`: `-0.426099`
    - `outdoor_humidity_pct`: `-0.025594`
    - `day_period`: `+0.842747`
    - `hour`: `-0.356397`
    - `minute`: `-0.001712`
- **Equation**:
  $$\text{Setpoint} = 41.8328 + (0.0116 \times T_{\text{in}}) - (0.0188 \times H_{\text{in}}) - (0.4261 \times T_{\text{out}}) - (0.0256 \times H_{\text{out}}) + (0.8427 \times \text{Period}) - (0.3564 \times \text{Hour}) - (0.0017 \times \text{Minute})$$

---

## 🛠️ Local Development

```bash
# Clone the repository
git clone https://github.com/Master-45-vic/SPIOT_Mini_Project_ML_Backend.git
cd SPIOT_Mini_Project_ML_Backend

# Install requirements
pip install -r requirements.txt

# Run tests
python test_api.py

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
