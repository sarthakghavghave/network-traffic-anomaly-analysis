import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import BEST_THRESHOLD
from utils import load_models, load_data, predict_window

iso, scaler, rf = load_models()
df_windowed, _, _, expected_cols = load_data()
feature_cols = [col for col in expected_cols if col not in ['window_id', 'window_attack']]

st.title("Replay Monitor")
st.markdown("Replaying UNSW-NB15 test set through the two-stage detection pipeline.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Windows",      len(df_windowed))
col2.metric("Actual Attacks",     int(df_windowed['window_attack'].sum()))
col3.metric("Stage 1 Flagged",    int(df_windowed['stage1_pred'].sum()))
col4.metric("After FP Reduction", int(df_windowed['final_pred'].sum()))

st.markdown("---")

c1, c2, c3 = st.columns([1, 1, 2])
speed     = c1.slider("Speed (windows/sec)", 1, 40, 10)
n_display = c2.slider("Windows to display",  20, 80, 50, 10)
start     = c3.button("Start Replay", type="primary", use_container_width=True)

st.markdown("---")

chart_col, table_col = st.columns([3, 2])
with chart_col:
    chart_placeholder = st.empty()
with table_col:
    table_placeholder = st.empty()
status_placeholder = st.empty()

if start:
    history = []

    for i, row in df_windowed.iterrows():
        window_row       = df_windowed.iloc[[i]]
        score, s1, s2    = predict_window(window_row, feature_cols, scaler, iso, rf)
        display_score    = round(-score, 4)
        actual           = int(row['window_attack'])

        if s2 == 1 and actual == 1:   verdict = 'Attack'
        elif s2 == 0 and actual == 0: verdict = 'Normal'
        elif s1 == 1 and s2 == 0:     verdict = 'FP Reduced'
        else:                          verdict = 'Missed'

        history.append({
            'window_id': int(row['window_id']),
            'score':     display_score,
            'stage1':    s1,
            'final':     s2,
            'actual':    actual,
            'verdict':   verdict
        })

        display    = history[-n_display:]
        df_display = pd.DataFrame(display)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_display['window_id'], y=df_display['score'],
            mode='lines', name='Anomaly Score',
            line=dict(color='royalblue', width=1.5)
        ))
        fig.add_hline(
            y=-BEST_THRESHOLD, line_dash='dash', line_color='orange',
            annotation_text=f'Threshold ({-BEST_THRESHOLD:.4f})'
        )
        attacks = df_display[df_display['final'] == 1]
        fig.add_trace(go.Scatter(
            x=attacks['window_id'], y=attacks['score'],
            mode='markers', name='Attack',
            marker=dict(color='red', size=6)
        ))
        fp_reduced = df_display[(df_display['stage1'] == 1) & (df_display['final'] == 0)]
        fig.add_trace(go.Scatter(
            x=fp_reduced['window_id'], y=fp_reduced['score'],
            mode='markers', name='FP Reduced',
            marker=dict(color='gold', size=8, symbol='star')
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation='h', y=-0.2),
            xaxis_title='Window ID',
            yaxis_title='Anomaly Score (higher = more suspicious)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        table_placeholder.dataframe(
            df_display[['window_id', 'score', 'verdict']]
            .tail(15).sort_values('window_id', ascending=False)
            .reset_index(drop=True),
            use_container_width=True, hide_index=True
        )

        fp_count      = sum(1 for h in history if h['stage1'] == 1 and h['final'] == 0)
        attack_count  = sum(1 for h in history if h['final'] == 1)
        status_placeholder.info(
            f"Window {int(row['window_id'])} / {len(df_windowed)} | "
            f"Attacks: {attack_count} | FPs Reduced: {fp_count}"
        )
        time.sleep(1 / speed)

    status_placeholder.success(
        f"Replay complete — {attack_count} attacks detected, {fp_count} FPs reduced"
    )