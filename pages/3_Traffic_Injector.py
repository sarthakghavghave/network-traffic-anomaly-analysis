# pages/2_Traffic_Injector.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import (PROCESSED_DIR, RAW_DIR, 
                    BEST_THRESHOLD, TOP_FEATURES)
from utils import load_models, load_data, predict_window

iso, scaler, rf = load_models()
df_windowed, df_normal, attack_stats, expected_cols = load_data()

feature_cols = [col for col in expected_cols if col not in ['window_id', 'window_attack']]

st.title("Traffic Injector")
st.markdown("""
Select an attack category, sample a real window from the dataset,
tweak key features, then run it through the detection pipeline.
""")

st.markdown("---")

st.subheader("Step 1 — Select Traffic Type")

category = st.selectbox(
    "Traffic Category",
    ["Normal", "Generic", "DoS", "Exploits", 
     "Fuzzers", "Reconnaissance", "Backdoor", 
     "Shellcode", "Worms", "Analysis"]
)

@st.cache_data
def load_raw_test():
    df = pd.read_csv(RAW_DIR / 'UNSW_NB15_testing-set.csv')
    df['window_id'] = df.index // 100
    return df

df_raw = load_raw_test()

@st.cache_data
def get_category_windows(category):
    if category == "Normal":
        mask = df_windowed['window_attack'] == 0
    else:
        cat_window_ids = df_raw[
            df_raw['attack_cat'] == category
        ]['window_id'].unique()
        mask = df_windowed['window_id'].isin(cat_window_ids)
    return df_windowed[mask]

category_windows = get_category_windows(category)

st.info(f"Found **{len(category_windows)}** windows of type **{category}**")

st.subheader("Step 2 — Sample a Real Window")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🎲 Sample Random Window", type="primary"):
        st.session_state['sampled_window'] = (
            category_windows.sample(1, random_state=None)
        )

if 'sampled_window' not in st.session_state:
    st.session_state['sampled_window'] = category_windows.sample(1)

sampled = st.session_state['sampled_window']

with col2:
    st.dataframe(
        sampled[TOP_FEATURES].T.rename(columns={sampled.index[0]: 'Value'}),
        use_container_width=True
    )

st.markdown("---")

st.subheader("Step 3 — Tweak Key Features")
st.markdown("Adjust sliders to modify the sampled window's key features:")

tweaked = sampled.copy()

cols = st.columns(3)
slider_configs = {
    'state_INT_mean':   (0.0,  1.0,  0.01),
    'proto_nunique':    (1,    35,   1),
    'sttl_mean':        (0.0,  254.0, 0.5),
    'sbytes_mean':      (0.0,  15.0,  0.1),
    'rate_mean':        (0.0,  12.0,  0.1),
    'dbytes_mean':      (0.0,  15.0,  0.1),
    'service_dns_mean': (0.0,  1.0,   0.01),
}

for i, (feat, (min_v, max_v, step)) in enumerate(slider_configs.items()):
    with cols[i % 3]:
        current_val = float(sampled[feat].values[0])
        current_val = np.clip(current_val, min_v, max_v)
        tweaked[feat] = st.slider(
            feat,
            min_value=float(min_v),
            max_value=float(max_v),
            value=float(current_val),
            step=float(step)
        )

st.markdown("---")

st.subheader("Step 4 — Run Through Pipeline")

if st.button("🚀 Detect", type="primary", use_container_width=True):
    score, stage1, stage2 = predict_window(
        tweaked, feature_cols, scaler, iso, rf
    )


    st.markdown("### Pipeline Decision")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("**Stage 1 — Isolation Forest**")
        if stage1 == 1:
            st.error(f"⚠️ Flagged as Anomaly\nScore: {score:.4f}")
        else:
            st.success(f"✅ Normal\nScore: {score:.4f}")

    with p2:
        st.markdown("**Stage 2 — Random Forest**")
        if stage1 == 0:
            st.info("⏭️ Skipped\n(Not flagged by Stage 1)")
        elif stage2 == 1:
            st.error("⚠️ Confirmed Attack")
        else:
            st.success("✅ False Positive Reduced")

    with p3:
        st.markdown("**Final Verdict**")
        if stage2 == 1:
            st.error("🔴 ATTACK DETECTED")
        elif stage1 == 1 and stage2 == 0:
            st.warning("🟡 FALSE POSITIVE REDUCED\nStage 2 cleared this alert")
        else:
            st.success("🟢 NORMAL TRAFFIC")

    st.markdown("### Anomaly Score")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=-score,
        gauge={
            'axis': {'range': [-0.25, 0.1]},
            'bar': {'color': "red" if stage1 == 1 else "green"},
            'steps': [
                {'range': [-0.25, -BEST_THRESHOLD],
                'color': "lightgreen"},
                {'range': [-BEST_THRESHOLD, 0.1],
                'color': "lightcoral"},
            ],
            'threshold': {
                'line': {'color': "orange", 'width': 4},
                'thickness': 0.75,
                'value': -BEST_THRESHOLD
            }
        },
        title={'text': "Anomaly Score (higher = more suspicious)"}
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)