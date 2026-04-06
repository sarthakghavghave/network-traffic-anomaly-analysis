import time
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import BEST_THRESHOLD, PROCESSED_DIR
from utils import load_models, load_data, predict_window

iso, scaler, rf = load_models()
_, _, _, expected_cols = load_data()
feature_cols = [col for col in expected_cols if col not in ['window_id', 'window_attack']]

BUFFER_FILE = PROCESSED_DIR / "live_buffer.json"

st.title("Live Monitor")
st.markdown("Scoring real network traffic captured from your network adapter.")
st.info(
    "Run `python scripts/live_monitor.py` as Administrator before connecting.\n\n"
    "Run `python scripts/simulate_attack.py` to trigger a test anomaly."
)

start = st.button("Connect", type="primary", use_container_width=True)
st.markdown("---")

m1, m2, m3 = st.columns(3)
m_rate    = m1.empty()
m_flags   = m2.empty()
m_attacks = m3.empty()

chart_col, table_col = st.columns([3, 2])
with chart_col:
    chart_placeholder = st.empty()
with table_col:
    table_placeholder = st.empty()

def read_buffer():
    if not BUFFER_FILE.exists():
        return []
    try:
        with open(BUFFER_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

if start:
    st.success("Connected. Waiting for traffic windows...")

    last_id      = -1
    history      = []
    total_flags  = 0
    total_attacks = 0

    while True:
        buffer = read_buffer()

        if buffer and buffer[-1]['window_id'] > last_id:
            new_windows = [w for w in buffer if w['window_id'] > last_id]

            for w in new_windows:
                df_row           = pd.DataFrame([w])
                score, s1, s2    = predict_window(df_row, feature_cols, scaler, iso, rf)

                if s1 == 1: total_flags   += 1
                if s2 == 1: total_attacks += 1

                if s2 == 1:             verdict = 'Attack'
                elif s1 == 1 and s2==0: verdict = 'FP Reduced'
                else:                   verdict = 'Normal'

                history.append({
                    'window_id': w['window_id'],
                    'rate':      round(w.get('raw_rate', 0), 1),
                    'score':     round(-score, 4),
                    'stage1':    s1,
                    'final':     s2,
                    'verdict':   verdict
                })

            last_id = buffer[-1]['window_id']
            history = history[-50:]
            df_display = pd.DataFrame(history)

            current_rate = buffer[-1].get('raw_rate', 0)
            m_rate.metric("Traffic Rate",   f"{current_rate:.1f} pkts/s")
            m_flags.metric("Stage 1 Flags", total_flags)
            m_attacks.metric("Attacks Detected", total_attacks)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_display['window_id'], y=df_display['score'],
                mode='lines+markers', name='Anomaly Score',
                line=dict(color='royalblue', width=2),
                marker=dict(size=4)
            ))
            fig.add_hline(
                y=-BEST_THRESHOLD, line_dash='dash', line_color='orange',
                annotation_text=f'Threshold ({round(-BEST_THRESHOLD, 4)})'
            )
            attacks_df = df_display[df_display['final'] == 1]
            if len(attacks_df) > 0:
                fig.add_trace(go.Scatter(
                    x=attacks_df['window_id'], y=attacks_df['score'],
                    mode='markers', name='Attack',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            fp_df = df_display[(df_display['stage1']==1) & (df_display['final']==0)]
            if len(fp_df) > 0:
                fig.add_trace(go.Scatter(
                    x=fp_df['window_id'], y=fp_df['score'],
                    mode='markers', name='FP Reduced',
                    marker=dict(color='gold', size=8, symbol='star')
                ))
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation='h', y=-0.2),
                yaxis_title='Anomaly Score (higher = more suspicious)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            table_placeholder.dataframe(
                df_display[['window_id', 'rate', 'score', 'verdict']]
                .tail(15).sort_values('window_id', ascending=False)
                .reset_index(drop=True),
                use_container_width=True, hide_index=True
            )

        time.sleep(1.0)