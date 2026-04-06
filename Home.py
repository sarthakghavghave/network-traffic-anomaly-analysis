import streamlit as st
from PIL import Image
from utils import load_models, load_data
from config import FIG_DIR, BEST_THRESHOLD, WINDOW_SIZE

st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("Network Traffic Anomaly Detection & False Positive Reduction")
st.markdown("""
*Welcome to the IDS Dashboard.*\n
This interactive tool allows you to replay network traffic windows through our two-stage anomaly detection pipeline, visualizing the anomaly scores and model predictions in real-time.
""")
df_windowed, _, _, _ = load_data()
image = Image.open(FIG_DIR / "anomaly_scores.png")
# pipeline
st.markdown("### Detection Pipeline")
st.markdown(f"""1. **Stage 1 - Isolation Forest**: Each window's features are passed to a pre-trained Isolation Forest model which outputs an anomaly score. Windows with scores below the threshold of **{BEST_THRESHOLD}** are flagged.""")

st.image(image, caption="Isolation Forest Anomaly Scores")

st.markdown(f"""2. **Stage 2 - Random Forest**: Flagged windows are then passed to a Random Forest classifier that determines whether the window is an attack or normal, aiming to reduce false positives from Stage 1.""")
# Dataset
st.markdown("### Dataset Overview")
st.markdown(f"We are working with a test set of **{len(df_windowed)} windows** derived from the **[UNSW NB-15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)**.")

iso, scaler, rf = load_models()
df_windowed, df_normal, attack_stats, expected_cols = load_data()

feature_cols = [col for col in expected_cols if col not in ['window_id', 'window_attack']]

st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=80)
st.sidebar.title("🛡️ IDS Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("**Pipeline Status**")
st.sidebar.success("✓ Models ready")
st.sidebar.info("Select a page to begin")
st.sidebar.markdown(f"**Threshold:** `{BEST_THRESHOLD}`")
st.sidebar.markdown(f"**Window Size:** `{WINDOW_SIZE} flows`")