"""
Model Performance page: detailed metrics and evaluation results.
Streamlit page: pages/4_Model_Performance.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.config import PROCESSED_DIR

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Model Performance Metrics")
st.markdown("Detailed performance evaluation of the two-stage detection pipeline.")

tab1, tab2, tab3 = st.tabs(["Overall Performance", "Category Breakdown", "Stage Comparison"])

with tab1:
    st.markdown("### Isolation Forest + Random Forest Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stage 1 Recall", "~97%")
    with col2:
        st.metric("FP Reduction", "88.6%")
    with col3:
        st.metric("Final Recall", "~97")
    with col4:
        st.metric("Final FP Rate", "<3%")
    
    st.markdown("---")
    
    eval_file = PROCESSED_DIR / 'eval_results.csv'
    if eval_file.exists():
        st.markdown("### Evaluation Results")
        df_eval = pd.read_csv(eval_file)
        st.dataframe(df_eval, use_container_width=True)
    else:
        st.info("`eval_results.csv` not found.")

with tab2:
    st.markdown("### Attack Category Detection Rates")
    
    attack_file = PROCESSED_DIR / 'attack_category_results.csv'
    if attack_file.exists():
        df_attack = pd.read_csv(attack_file)
        
        st.dataframe(df_attack, use_container_width=True)
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                plot_df = df_attack.copy()
                if 'dominant_cat' in plot_df.columns:
                    plot_df = plot_df.rename(columns={'dominant_cat': 'category'})
                
                available_metrics = [c for c in ['detection_rate', 'precision', 'recall', 'f1_score'] if c in plot_df.columns]
                
                fig = px.bar(
                    plot_df,
                    x='category' if 'category' in plot_df.columns else plot_df.columns[0],
                    y=available_metrics if available_metrics else plot_df.columns[1:],
                    barmode='group',
                    title='Detection Metrics by Category',
                    labels={'value': 'Score', 'variable': 'Metric'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render chart: {e}")
        
        with col2:
            st.markdown("### Difficulty Ranking")
            sort_col = 'detection_rate' if 'detection_rate' in df_attack.columns else (
                'recall' if 'recall' in df_attack.columns else df_attack.columns[0]
            )
            df_sorted = df_attack.sort_values(sort_col, ascending=False)
            
            for idx, row in df_sorted.head(5).iterrows():
                metric = row.get('detection_rate', row.get('recall', row.get('f1_score', 0)))
                if metric > 1.0: 
                    display_metric = metric / 100.0
                    metric_val = metric
                else:
                    display_metric = metric
                    metric_val = metric * 100.0
                
                emoji = "🟢" if display_metric > 0.9 else "🟡" if display_metric > 0.7 else "🔴"
                category = row.get('dominant_cat', row.get('attack_category', 'Unknown'))
                
                st.metric(f"{emoji} {category}", f"{display_metric:.1%}")
    else:
        st.info("`attack_category_results.csv` not found")

with tab3:
    st.markdown("### Stage 1 vs Stage 2: Why Two Stages?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Stage 1: Isolation Forest (Unsupervised)")
        st.markdown("""
        **Function**: Detect anomalies broadly
        
        **Performance**:
        - Recall: ~97% (catches most attacks)
        - Precision: ~18% (many false alerts)
        
        **Why**: No labels needed, catches novel attacks
        """)
    
    with col2:
        st.markdown("#### Stage 2: Random Forest (Supervised)")
        st.markdown("""
        **Function**: Filter false positives
        
        **Performance**:
        - Reduces FP by 88.6%
        - Maintains Recall: ~97%
        
        **Why**: Refines Stage 1 using attack examples
        """)
    
    st.markdown("---")
    
    st.markdown("#### The Problem & Solution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Single Detector Problem**:")
        st.markdown("""
        Too strict → misses attacks (low recall)
        Too sensitive → high false alerts (low precision)
        Can't balance both simultaneously
        **Result**: Operationally unusable
        """)
    
    with col2:
        st.markdown("**Two-Stage Solution**:")
        st.markdown("""
        Stage 1: Broad net (97% recall)
        Stage 2: Refine (88.6% FP reduction)
        Best of both worlds
        **Result**: Operationally viable
        """)
