import streamlit as st
import joblib
import pandas as pd
import json
from scripts.config import MODEL_DIR, PROCESSED_DIR

@st.cache_resource
def load_pipeline():
    """Loads the requested detection pipeline."""
    iso = joblib.load(MODEL_DIR / 'isolation_forest.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    rf = joblib.load(MODEL_DIR / 'rf_stage2.pkl')

    return iso, scaler, rf, 0.0262

@st.cache_data
def load_data():
    df_windowed  = pd.read_csv(PROCESSED_DIR / 'test_windowed_results.csv')
    df_normal    = pd.read_csv(PROCESSED_DIR / 'normal_windowed.csv')
    attack_stats = pd.read_csv(PROCESSED_DIR / 'attack_category_results.csv')
    
    with open(PROCESSED_DIR / 'windowed_columns.json') as f:
        expected_cols = json.load(f)
    
    return df_windowed, df_normal, attack_stats, expected_cols

def predict_window(window_features, feature_cols, scaler, s1_model, s2_model):
    X = window_features[feature_cols].values

    X_scaled = scaler.transform(X)

    raw_score = s1_model.decision_function(X_scaled)[0]

    score = -raw_score
    threshold = -0.0262

    s1 = int(raw_score < 0.0262)

    s2 = s1
    if s1 == 1:
        s2 = int(s2_model.predict(X)[0])

    return score, threshold, s1, s2