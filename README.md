# Traffic Demand Prediction API

A machine learning REST API that predicts traffic demand for a given geohash location and time, built using LightGBM and deployed on Render.

## Features
- Feature engineering on time, road type, geohash hierarchy, and weather
- LightGBM regression model
- FastAPI backend with auto-generated docs
- Deployed live on Render

## Tech Stack
Python, LightGBM, FastAPI, Pandas, NumPy, Render

## Usage
POST to `/predict` with JSON:
```json
{
  "geohash": "tdr1x2",
  "timestamp": "14:30",
  "RoadType": "Highway",
  "NumberofLanes": 4,
  "LargeVehicles": "Yes",
  "Landmarks": "Mall",
  "Weather": "Clear"
}
```