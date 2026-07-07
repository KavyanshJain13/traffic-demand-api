import pandas as pd
import numpy as np
import lightgbm as lgb
import pygeohash as pgh
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Loading dataset...")
train = pd.read_csv('dataset/train.csv')

# =====================================================
# Feature Engineering (same logic as your original code)
# =====================================================

def engineer_features(df, category_maps=None, fit=True):
    df = df.copy()

    df[['hour', 'minute']] = (
        df['timestamp'].str.split(':', expand=True).astype(float)
    )
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

    cols_to_drop = ['timestamp', 'Index']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    cat_cols = [
        'geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
        'road_lane', 'road_hour', 'geo_3', 'geo_4', 'geo_5', 'quarter_hour'
    ]

    if fit:
        category_maps = {}

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            if fit:
                categories = sorted(df[col].unique().tolist())
                category_maps[col] = categories
            else:
                categories = category_maps[col]
            df[col] = pd.Categorical(df[col], categories=categories)

    return df, category_maps

print("Engineering features...")
y = train['demand']
X, category_maps = engineer_features(train.drop(columns=['demand']), fit=True)

feature_cols = X.columns.tolist()

# =====================================================
# Train ONE final model on all data (simple + fast for demo)
# =====================================================

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 45,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}

print("Training model...")
train_data = lgb.Dataset(X, label=y)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=800
)

# =====================================================
# Save everything the app will need
# =====================================================

joblib.dump({
    'model': model,
    'category_maps': category_maps,
    'feature_cols': feature_cols
}, 'model_artifacts.pkl')

print("Saved model_artifacts.pkl. Training complete!")