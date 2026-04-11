import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_pipeline, load_data, predict_window

st.set_page_config(page_title="Replay Monitor", page_icon="📈", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Replay Monitor")
st.markdown("Replaying UNSW-NB15 test set through the selected two-stage pipeline.")

# Sidebar Configuration
st.sidebar.header("🛠️ Configuration")
pipeline_type = st.sidebar.selectbox(
    "Detection Pipeline",
    ["Isolation Forest + RF", "Autoencoder + SVM"],
    index=1
)

# Load resources
s1_model, scaler, s2_model, model_threshold = load_pipeline(pipeline_type)
df_windowed, _, _, expected_cols = load_data()
feature_cols = [col for col in expected_cols if col not in ['window_id', 'window_attack']]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Windows",      len(df_windowed))
col2.metric("Actual Attacks",     int(df_windowed['window_attack'].sum()))
s1_placeholder = col3.empty()
s2_placeholder = col4.empty()

st.markdown("---")

c1, c2, c3 = st.columns([1, 1, 2])
speed     = c1.slider("Replay Speed", 1, 40, 15)
n_display = c2.slider("History Window", 20, 100, 50)
start     = c3.button("Start Replay Simulation", type="primary", use_container_width=True)

st.markdown("---")

chart_col, table_col = st.columns([3, 2])
with chart_col:
    chart_placeholder = st.empty()
with table_col:
    table_placeholder = st.empty()
status_placeholder = st.empty()

if start:
    history = []
    
    # Reset metrics counters
    s1_flags = 0
    s2_alerts = 0

    for i, row in df_windowed.iterrows():
        window_row = df_windowed.iloc[[i]]
        score, threshold, s1, s2 = predict_window(window_row, feature_cols, scaler, s1_model, s2_model, pipeline_type)
        
        actual = int(row['window_attack'])
        
        # Determine verdict for display
        if s2 == 1 and actual == 1:   
            verdict = 'Attack (Detected)'
            s2_alerts += 1
            s1_flags += 1
        elif s2 == 0 and actual == 0: 
            verdict = 'Normal'
        elif s1 == 1 and s2 == 0:     
            verdict = 'False Alarm (Blocked)'
            s1_flags += 1
        elif s1 == 0 and actual == 1: 
            verdict = 'Missed'
        else:
            verdict = 'Normal'

        history.append({
            'window_id': int(row['window_id']),
            'score':     round(score, 4),
            'stage1':    s1,
            'final':     s2,
            'actual':    actual,
            'verdict':   verdict
        })

        display    = history[-n_display:]
        df_display = pd.DataFrame(display)

        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_display['window_id'], y=df_display['score'],
            mode='lines', name='Suspicion Score',
            line=dict(color='#007BFF', width=2)
        ))
        fig.add_hline(
            y=threshold, line_dash='dash', line_color='#FFC107',
            annotation_text='Threshold', annotation_position="top left"
        )
        
        # Mark alerts
        alerts = df_display[df_display['final'] == 1]
        fig.add_trace(go.Scatter(
            x=alerts['window_id'], y=alerts['score'],
            mode='markers', name='Alert Triggered',
            marker=dict(color='#DC3545', size=10, symbol='circle')
        ))
        
        # Mark FP Reductions
        fp_red = df_display[(df_display['stage1'] == 1) & (df_display['final'] == 0)]
        fig.add_trace(go.Scatter(
            x=fp_red['window_id'], y=fp_red['score'],
            mode='markers', name='FP Blocked',
            marker=dict(color='#28A745', size=12, symbol='star')
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation='h', y=-0.2),
            xaxis_title='Window ID',
            yaxis_title='Suspicion Score',
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Result table
        table_placeholder.dataframe(
            df_display[['window_id', 'score', 'verdict']]
            .tail(15).sort_values('window_id', ascending=False)
            .reset_index(drop=True),
            use_container_width=True, hide_index=True
        )

        # Update metrics
        s1_placeholder.metric("Stage 1 Flags", s1_flags)
        s2_placeholder.metric("Final Alerts",  s2_alerts)
        
        status_placeholder.info(
            f"Analyzing Window {int(row['window_id'])} | "
            f"Pipeline: {pipeline_type}"
        )
        time.sleep(1 / speed)

    status_placeholder.success(
        f"Replay complete! Detected {s2_alerts} attacks and blocked {s1_flags - s2_alerts} false alarms."
    )
