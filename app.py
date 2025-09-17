from flask import Flask, request, jsonify
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import DBSCAN
import numpy as np
import os
import requests
import time
import datetime
import pandas as pd
import joblib
import sklearn.compose._column_transformer as ct  # 👈 needed for patch

# -------------------------
# Patch missing class (_RemainderColsList)
# -------------------------
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

# -------------------------
# Flask app setup
# -------------------------
app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# -------------------------
# File Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INCIDENTS_FILE = os.path.join(BASE_DIR, "synthetic_incidents.geojson")
CRIME_FILE = os.path.join(BASE_DIR, "crime_dataset_filtered.json")
MODEL_FILE = os.path.join(BASE_DIR, "my_pipeline.pkl")

# -------------------------
# Load model pipeline
# -------------------------
try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    raise FileNotFoundError(f"❌ Could not load model file: {MODEL_FILE}\n{e}")

# -------------------------
# Load incidents & crime dataset
# -------------------------
try:
    incidents = gpd.read_file(INCIDENTS_FILE).set_crs("EPSG:4326")
except Exception as e:
    raise FileNotFoundError(f"❌ Could not load incidents file: {INCIDENTS_FILE}\n{e}")

try:
    crime_df = pd.read_json(CRIME_FILE)
except Exception as e:
    raise FileNotFoundError(f"❌ Could not load crime dataset: {CRIME_FILE}\n{e}")

coords = np.array([[p.x, p.y] for p in incidents.geometry])
db = DBSCAN(eps=0.05, min_samples=3).fit(coords)
incidents["cluster"] = db.labels_

hotspot_polys = []
for cluster_id in set(db.labels_):
    if cluster_id == -1:
        continue
    cluster_points = incidents[incidents["cluster"] == cluster_id].geometry
    poly = MultiPoint(cluster_points).buffer(0.05).convex_hull
    hotspot_polys.append({"cluster": cluster_id, "geometry": poly})

hotspots = gpd.GeoDataFrame(hotspot_polys, crs="EPSG:4326")
risk_scores = incidents.groupby("cluster").size().reset_index(name="INCIDENTS")
hotspots = hotspots.merge(risk_scores, on="cluster", how="left")
hotspots["RISK_INDEX"] = (hotspots["INCIDENTS"] / hotspots["INCIDENTS"].max()) * 10

# -------------------------
# Risk Calculation
# -------------------------
def compute_risk(lat, lon):
    user_point = Point(lon, lat)
    for _, row in hotspots.iterrows():
        if row.geometry.contains(user_point):
            return row["RISK_INDEX"]

    # Project to metric CRS for distance calculation
    incidents_proj = incidents.to_crs(epsg=3857)
    user_point_proj = gpd.GeoSeries([user_point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    incidents_proj['distance_to_user'] = incidents_proj.geometry.distance(user_point_proj)
    min_dist = incidents_proj['distance_to_user'].min()

    C, k = 10000, 1000
    risk = C / (min_dist + k)
    return min(round(risk, 2), 10)

# -------------------------
# Weather & Temporal
# -------------------------
API_KEY = "8f1c266212dee5e4d5e29c10f24e6ae2"

def get_district_from_lat_lon(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    headers = {"User-Agent": "MyApp/3.0 (div2020123@gmail.com)"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        try:
            data = response.json()
            return data.get("address", {}).get("county") or data.get("address", {}).get("state_district")
        except ValueError:
            return None
    return None

def get_weather_and_time(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return {}

    data = response.json()
    temp = data['main']['temp']
    visibility = data.get('visibility', 0) / 1000
    condition = data['weather'][0]['main']

    # Season
    month = datetime.datetime.now().month
    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Autumn"

    # Time of day
    hour = time.localtime().tm_hour
    if 5 <= hour < 12:
        time_of_day = "Morning"
    elif 12 <= hour < 17:
        time_of_day = "Afternoon"
    elif 17 <= hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"

    day_of_week = datetime.datetime.now().strftime("%A")

    return {
        "season": season,
        "weather": condition,
        "temperature": temp,
        "visibility": visibility,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "month": month
    }

# -------------------------
# Socio-crime data
# -------------------------
def get_socio_crime_data(district, lat=None, lon=None):
    matching_rows = crime_df[crime_df['district'].str.lower().str.contains(district.lower(), na=False)]
    if not matching_rows.empty:
        row = matching_rows.iloc[0].to_dict()
    else:
        avg_values = crime_df.drop(columns=["district", "state"], errors="ignore").mean(numeric_only=True).to_dict()
        row = {**avg_values}
        row["district"] = district
        row["state"] = "Unknown"
    if lat is not None and lon is not None:
        weather_time = get_weather_and_time(lat, lon)
        row.update(weather_time)
    return row

# -------------------------
# Flask Endpoints
# -------------------------
# -------------------------
# Flask Endpoints
# -------------------------

@app.route('/get_full_context', methods=['GET'])
def get_full_context():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
    except:
        return jsonify({"error": "Invalid or missing lat/lon"}), 400

    # ---- Risk + context ----
    risk_index = compute_risk(lat, lon)
    weather_time = get_weather_and_time(lat, lon)
    district = get_district_from_lat_lon(lat, lon)
    if not district:
        return jsonify({"error": "Unable to determine district"}), 400

    socio_crime = get_socio_crime_data(district, lat=lat, lon=lon)

    features = {
        "risk_index": risk_index,
        "district": district,
        "latitude": lat,
        "longitude": lon,
        **weather_time,
        **socio_crime
    }

    input_df = pd.DataFrame([features])

    # ---- Prediction ----
    pred_class = model.predict(input_df)[0]

    # ✅ Probabilities
    prob_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        class_labels = model.classes_  # already contains your string labels if trained that way
        prob_dict = {str(label): float(prob) for label, prob in zip(class_labels, probs)}

    # -------------------------
    # ✅ Fuse with risk_index
    # -------------------------
    fused_probs = prob_dict.copy()
    alpha = 0.7  # ML model weight
    beta = 1 - alpha  # risk_index weight

    # Normalize risk_index to [0,1]
    r = min(max(risk_index / 10.0, 0), 1)

    # Map risk_index → extra weight for risky classes (1 and 2 here)
    for label in fused_probs.keys():
        p_model = prob_dict[label]
        if label in ["1", "2"]:  # risky classes
            p_final = alpha * p_model + beta * r
        else:  # safe/neutral classes
            p_final = alpha * p_model + beta * (1 - r)
        fused_probs[label] = p_final

    # Renormalize
    total = sum(fused_probs.values())
    fused_probs = {k: v / total for k, v in fused_probs.items()}

    # Final prediction
    final_prediction = max(fused_probs, key=fused_probs.get)

    # -------------------------
    # ✅ Return everything
    # -------------------------
    return jsonify({
        "input_features": features,
        "prediction": str(pred_class),   # raw model output
        "probabilities": prob_dict,      # raw model probabilities
        "final_probabilities": fused_probs,  # risk-aware probabilities
        "final_prediction": final_prediction  # risk-aware prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
