import os
import joblib
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
from datetime import datetime
from difflib import get_close_matches

from .content import VALID_AREAS

# ======================================================
# PATH SETUP
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_files", "traffic_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model_files", "encoders.pkl")
AREA_STATS_PATH = os.path.join(BASE_DIR, "model_files", "area_stats.pkl")

# ======================================================
# ⭐ LAZY MODEL LOADING (FAST START)
# ======================================================

model = None
encoder = None
features = None
area_stats = None
cat_feats = None
MODEL_AREAS = []

def load_model():
    global model, encoder, features, area_stats, cat_feats, MODEL_AREAS

    if model is None:
        model_data = joblib.load(MODEL_PATH)
        encoder_data = joblib.load(ENCODER_PATH)
        area_stats_data = joblib.load(AREA_STATS_PATH)

        model = model_data["model"]
        features = model_data["features"]
        encoder = encoder_data["ordinal_encoder"]
        cat_feats = encoder_data["cat_features"]
        area_stats = area_stats_data

        MODEL_AREAS = list(encoder.categories_[0])

# ======================================================
# SPELLING NORMALIZATION
# ======================================================

def normalize_area_name(user_area):
    load_model()

    if not user_area:
        return None

    user_area = user_area.strip().lower()

    for area in VALID_AREAS + MODEL_AREAS:
        if area.lower() == user_area:
            return area

    match = get_close_matches(
        user_area,
        VALID_AREAS + MODEL_AREAS,
        n=1,
        cutoff=0.6
    )
    return match[0] if match else None


# ======================================================
# TRAFFIC PREDICTION
# ======================================================

TRAFFIC_CACHE = {}

def predict_traffic_internal(area, timestamp=None):

    load_model()   # ⭐ IMPORTANT

    now = datetime.now()
    cache_key = f"{area}_{now.strftime('%Y-%m-%d_%H')}"

    if cache_key in TRAFFIC_CACHE:
        return TRAFFIC_CACHE[cache_key]

    stats = area_stats.get(area, {"mean": 50})
    base = stats["mean"]

    if 7 <= now.hour <= 9:
        base += 15
    elif 17 <= now.hour <= 20:
        base += 18
    elif 22 <= now.hour or now.hour <= 5:
        base -= 10

    vehicles = int(base * 2)
    avg_speed = max(15, 60 - base)

    lag_1 = base
    lag_24 = base * 0.95
    lag_168 = base * 0.9

    row = {
        "hour": now.hour,
        "dayofweek": now.weekday(),
        "month": now.month,
        "is_weekend": int(now.weekday() >= 5),
        "is_morning_rush": int(7 <= now.hour <= 9),
        "is_evening_rush": int(17 <= now.hour <= 20),
        "vehicles": vehicles,
        "avg_speed_kmph": avg_speed,
        "temperature_c": 30,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "roll_mean_3": (lag_1 + lag_24 + lag_168) / 3,
        "roll_std_3": np.std([lag_1, lag_24, lag_168]),
    }

    cat_df = pd.DataFrame([[area, "Clear"]], columns=cat_feats).astype(str)
    encoded = encoder.transform(cat_df)

    for i, c in enumerate(cat_feats):
        row[c] = encoded[0, i]

    X = pd.DataFrame([row])[features]
    dmatrix = xgb.DMatrix(X)

    prediction = float(model.predict(dmatrix)[0])
    prediction = max(15, min(prediction, 80))
    prediction = round(prediction, 2)

    TRAFFIC_CACHE[cache_key] = prediction
    return prediction


# ======================================================
# WEATHER API
# ======================================================

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

def get_weather_data(lat, lon):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        data = requests.get(url, params=params, timeout=10).json()

        return {
            "temperature": data["main"]["temp"],
            "condition": data["weather"][0]["main"],
            "rain": data.get("rain", {}).get("1h", 0)
        }

    except:
        return {"temperature": 30, "condition": "Clear", "rain": 0}


# ======================================================
# SIMPLE BEST ROUTE
# ======================================================

def best_route(data):
    start = data.get("start")
    end = data.get("end")
    mode = data.get("mode", "car")

    if not start or not end:
        return {"error": "start and end required"}

    return {
        "message": "Best route running",
        "start": start,
        "end": end,
        "mode": mode
    }


# ======================================================
# CATALYST MAIN HANDLER
# ======================================================

def handler(request, response):

    try:
        data = request.get_json() or {}
    except:
        data = {}

    path = request.path or "/"

    # ---------- PREDICT ----------
    if path == "/predict":

        raw_area = data.get("area")

        if not raw_area:
            response.status(400)
            response.write({"error": "area required"})
            return

        area = normalize_area_name(raw_area)

        if not area:
            response.status(400)
            response.write({
                "error": "Unknown area",
                "suggestions": VALID_AREAS[:5]
            })
            return

        traffic = predict_traffic_internal(area)

        response.write({
            "area": area,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "predicted_traffic_percentage": traffic
        })
        return

    # ---------- BEST ROUTE ----------
    elif path == "/best-route":

        result = best_route(data)
        response.write(result)
        return

    # ---------- DEFAULT ----------
    response.write({"message": "Traffic Backend Running"})