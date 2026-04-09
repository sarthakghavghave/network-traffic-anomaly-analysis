import streamlit as st
import joblib
import pandas as pd
import json
from config import MODEL_DIR, PROCESSED_DIR, BEST_THRESHOLD

@st.cache_resource
def load_models():
    iso = joblib.load(MODEL_DIR / 'isolation_forest.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    rf = joblib.load(MODEL_DIR / 'rf_stage2.pkl')
    return iso, scaler, rf

@st.cache_data
def load_data():
    df_windowed  = pd.read_csv(PROCESSED_DIR / 'test_windowed_results.csv')
    df_normal    = pd.read_csv(PROCESSED_DIR / 'normal_windowed.csv')
    attack_stats = pd.read_csv(PROCESSED_DIR / 'attack_category_results.csv')
    
    with open(PROCESSED_DIR / 'windowed_columns.json') as f:
        expected_cols = json.load(f)
    
    return df_windowed, df_normal, attack_stats, expected_cols

def predict_window(window_features, feature_cols, scaler, iso, rf):
    scaled = scaler.transform(window_features[feature_cols])

    score  = iso.decision_function(scaled)[0]
    stage1 = int(score < BEST_THRESHOLD)

    stage2 = stage1
    if stage1 == 1:
        stage2 = int(rf.predict(window_features[feature_cols])[0])

    return score, stage1, stage2