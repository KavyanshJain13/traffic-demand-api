import pandas as pd
import numpy as np
import pygeohash as pgh
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Traffic Demand Prediction API")

artifacts = joblib.load('model_artifacts.pkl')
model = artifacts['model']
category_maps = artifacts['category_maps']
feature_cols = artifacts['feature_cols']


class TrafficInput(BaseModel):
    geohash: str
    timestamp: str          # format "HH:MM"
    RoadType: str
    NumberofLanes: int
    LargeVehicles: str
    Landmarks: str
    Weather: str


def engineer_features_single(data: dict):
    df = pd.DataFrame([data])

    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(float)
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    df['time_sin'] = np.sin(2 * np.pi * df['total_minutes'] / 1440)
    df['time_cos'] = np.cos(2 * np.pi * df['total_minutes'] / 1440)
    df['quarter_hour'] = (df['hour'] * 4 + (df['minute'] // 15))

    df['road_lane'] = df['RoadType'].astype(str) + "_" + df['NumberofLanes'].astype(str)
    df['road_hour'] = df['RoadType'].astype(str) + "_" + df['hour'].astype(int).astype(str)

    df['geo_3'] = df['geohash'].str[:3]
    df['geo_4'] = df['geohash'].str[:4]
    df['geo_5'] = df['geohash'].str[:5]

    decoded = df['geohash'].apply(lambda x: pgh.decode(x))
    df['lat'] = decoded.apply(lambda x: x[0])
    df['lon'] = decoded.apply(lambda x: x[1])

    df = df.drop(columns=['timestamp'])

    cat_cols = [
        'geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
        'road_lane', 'road_hour', 'geo_3', 'geo_4', 'geo_5', 'quarter_hour'
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            df[col] = pd.Categorical(df[col], categories=category_maps[col])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]
    return df


@app.get("/")
def home():
    return {"message": "Traffic Demand Prediction API is running!"}


@app.post("/predict")
def predict(input_data: TrafficInput):
    features = engineer_features_single(input_data.dict())
    prediction = model.predict(features)
    return {"predicted_demand": float(prediction[0])}
