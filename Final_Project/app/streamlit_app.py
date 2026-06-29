"""
=============================================================================
 AI PERSONALIZED LEARNING AGENT -- STREAMLIT DASHBOARD
=============================================================================
 Interactive UI for students and teachers to:
 - Select a student and view their profile
 - Run the AI agent pipeline
 - View personalized study plans, resources, and progress reports
=============================================================================
"""

import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

import pickle
import torch

from src.knowledge_tracer.dkt_model import load_dkt_model, predict_mastery, sequences_to_tensors
from src.gap_detector.xgboost_model import load_gap_detector, identify_weak_areas
from src.recommender.collab_filter import load_recommender
from src.study_planner.planner import generate_study_plan
from src.progress_reporter.reporter import generate_progress_report

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI Learning Agent",
    page_icon="&#127891;",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }
    
    h1, h2, h3 { color: #e0e0ff !important; }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #a0a0c0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        color: #a0a0c0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    .risk-badge-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .risk-badge-low {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .resource-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    
    .resource-card:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .plan-day {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    
    .weak-area-high { border-left: 4px solid #ff6b6b; }
    .weak-area-medium { border-left: 4px solid #f39c12; }
    .weak-area-low { border-left: 4px solid #2ecc71; }
    
    .stSidebar { background: rgba(15, 12, 41, 0.95) !important; }
    
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD DATA & MODELS
# =============================================================================
@st.cache_data
def load_data():
    """Load processed student data and configurations."""
    data_dir = os.path.join(PROJECT_DIR, "data", "processed")
    df = pd.read_csv(os.path.join(data_dir, "features_engineered.csv"))
    
    with open(os.path.join(data_dir, "feature_config.pkl"), 'rb') as f:
        config = pickle.load(f)
    
    with open(os.path.join(data_dir, "dkt_sequences.pkl"), 'rb') as f:
        sequences = pickle.load(f)
    
    return df, config, sequences


@st.cache_resource
def load_models():
    """Load all trained models."""
    models_dir = os.path.join(PROJECT_DIR, "models")
    
    dkt_model = load_dkt_model(os.path.join(models_dir, "dkt_model.pth"))
    gap_model = load_gap_detector(os.path.join(models_dir, "gap_detector.pkl"))
    recommender = load_recommender(os.path.join(models_dir, "recommender.pkl"))
    
    return dkt_model, gap_model, recommender


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_mastery_radar(mastery_scores):
    """Create a radar chart for mastery levels."""
    concepts = list(mastery_scores.keys())
    values = [mastery_scores[c] * 100 for c in concepts]
    values.append(values[0])  # Close the polygon
    concepts.append(concepts[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=concepts,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8, color='#667eea'),
        name='Mastery'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                          tickfont=dict(color='#a0a0c0', size=10),
                          gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(tickfont=dict(color='#e0e0ff', size=12),
                           gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=30, b=30)
    )
    return fig


def create_grade_progression(g1, g2, g3):
    """Create a grade progression line chart."""
    fig = go.Figure()
    
    periods = ['Period 1 (G1)', 'Period 2 (G2)', 'Final (G3)']
    grades = [g1, g2, g3]
    
    # Gradient color based on trend
    color = '#2ecc71' if g3 >= g1 else '#ff6b6b'
    
    fig.add_trace(go.Scatter(
        x=periods, y=grades,
        mode='lines+markers+text',
        text=[str(g) for g in grades],
        textposition='top center',
        textfont=dict(color='#e0e0ff', size=14, family='Inter'),
        line=dict(color=color, width=3),
        marker=dict(size=14, color=color, line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor=f'rgba({",".join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0,2,4))}, 0.1)'
    ))
    
    # Pass line
    fig.add_hline(y=10, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Pass Line (10)", annotation_font_color="#a0a0c0")
    
    fig.update_layout(
        yaxis=dict(range=[0, 22], title='Grade', color='#a0a0c0',
                   gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(color='#a0a0c0', gridcolor='rgba(255,255,255,0.05)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )
    return fig


def create_class_distribution(df):
    """Create a distribution chart of all students."""
    fig = px.histogram(
        df, x='G3', nbins=20,
        color_discrete_sequence=['#667eea'],
        labels={'G3': 'Final Grade', 'count': 'Students'}
    )
    fig.add_vline(x=10, line_dash="dash", line_color="#ff6b6b",
                  annotation_text="Pass/Fail", annotation_font_color="#ff6b6b")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='#a0a0c0', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(color='#a0a0c0', gridcolor='rgba(255,255,255,0.05)'),
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )
    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Load data and models
    try:
        df, config, sequences = load_data()
        dkt_model, gap_model, recommender = load_models()
        feature_cols = config['feature_cols']
        resources = config['resources']
    except Exception as e:
        st.error(f"Error loading data/models: {e}")
        st.info("Please run `python run_pipeline.py` first to train models.")
        return
    
    # --- HEADER ---
    st.markdown('<div class="hero-title">&#127891; AI Personalized Learning Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Powered by Deep Knowledge Tracing + XGBoost + Collaborative Filtering</div>', unsafe_allow_html=True)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### &#127919; Student Selection")
        
        # Student selector
        student_id = st.selectbox(
            "Choose a student",
            options=range(len(df)),
            format_func=lambda x: f"Student #{x} (G3={df.iloc[x]['G3']:.0f}, "
                                  f"{'At-Risk' if df.iloc[x].get('at_risk', 0) == 1 else 'Pass'})"
        )
        
        st.markdown("---")
        st.markdown("### &#128202; Quick Filters")
        
        show_at_risk = st.checkbox("Show at-risk only", value=False)
        if show_at_risk:
            at_risk_ids = df[df['at_risk'] == 1].index.tolist()
            student_id = st.selectbox("At-risk students", at_risk_ids,
                                      format_func=lambda x: f"Student #{x} (G3={df.iloc[x]['G3']:.0f})")
        
        st.markdown("---")
        st.markdown("### &#128218; Dataset Overview")
        st.metric("Total Students", len(df))
        st.metric("At-Risk Students", int(df['at_risk'].sum()))
        st.metric("Pass Rate", f"{(1-df['at_risk'].mean())*100:.1f}%")
        
        st.markdown("---")
        run_agent = st.button("&#9889; Run AI Agent", use_container_width=True)
    
    # --- STUDENT PROFILE ---
    student = df.iloc[student_id]
    
    st.markdown("## Student Profile")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Student ID", f"#{student_id}")
    with col2:
        st.metric("Age", int(student['age']))
    with col3:
        st.metric("School", student['school'])
    with col4:
        st.metric("Final Grade", f"{student['G3']:.0f}/20")
    with col5:
        risk_status = "At Risk" if student.get('at_risk', 0) == 1 else "On Track"
        st.metric("Status", risk_status)
    
    # Grade Progression
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### Grade Progression")
        fig = create_grade_progression(student['G1'], student['G2'], student['G3'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("### Class Distribution")
        fig = create_class_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- RUN AGENT ---
    if run_agent:
        st.markdown("---")
        st.markdown("## &#129302; AI Agent Results")
        
        with st.spinner("Running AI Agent pipeline..."):
            # Tool 1: Knowledge Tracer
            mastery_list = predict_mastery(dkt_model, [sequences[student_id]], n_concepts=5)
            mastery = mastery_list[0]
            
            # Tool 2: Gap Detector
            student_features = df.iloc[student_id][feature_cols].values.astype(float)
            student_features = np.nan_to_num(student_features, nan=0, posinf=0, neginf=0)
            gap_result = identify_weak_areas(gap_model, student_features, feature_cols)
            
            # Tool 3: Recommender
            recs = recommender.recommend(student_id, gap_result['weak_areas'], top_n=3)
            
            # Tool 4: Study Planner
            plan = generate_study_plan(student_id, mastery, gap_result['weak_areas'],
                                       recs, student.to_dict())
            
            # Tool 5: Progress Report
            report = generate_progress_report(
                student_id, student.to_dict(), mastery,
                gap_result['weak_areas'], gap_result, plan
            )
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "&#129504; Knowledge Mastery",
            "&#128269; Gap Analysis",
            "&#128218; Resources",
            "&#128197; Study Plan",
            "&#128202; Progress Report"
        ])
        
        # TAB 1: MASTERY
        with tab1:
            col_radar, col_details = st.columns([1, 1])
            
            with col_radar:
                st.markdown("#### Concept Mastery Radar")
                fig = create_mastery_radar(mastery)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_details:
                st.markdown("#### Mastery Breakdown")
                for concept, score in mastery.items():
                    pct = score * 100
                    color = '#2ecc71' if pct >= 70 else '#f39c12' if pct >= 40 else '#ff6b6b'
                    st.markdown(f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #e0e0ff; font-weight: 500;">{concept.title()}</span>
                            <span style="color: {color}; font-weight: 600;">{pct:.1f}%</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); border-radius: 8px; height: 8px; overflow: hidden;">
                            <div style="background: {color}; height: 100%; width: {pct}%; border-radius: 8px; transition: width 0.5s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                avg_mastery = np.mean(list(mastery.values())) * 100
                st.markdown(f"""
                <div style="margin-top: 20px; padding: 16px; background: rgba(102,126,234,0.1); border-radius: 12px; border: 1px solid rgba(102,126,234,0.2);">
                    <div style="color: #a0a0c0; font-size: 0.85rem;">Overall Mastery</div>
                    <div style="color: #667eea; font-size: 1.8rem; font-weight: 700;">{avg_mastery:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        # TAB 2: GAP ANALYSIS
        with tab2:
            risk_prob = gap_result['risk_probability']
            is_risk = gap_result['is_at_risk']
            
            # Risk meter
            col_risk, col_areas = st.columns([1, 1])
            
            with col_risk:
                st.markdown("#### Risk Assessment")
                badge_class = "risk-badge-high" if is_risk else "risk-badge-low"
                badge_text = "AT RISK" if is_risk else "ON TRACK"
                st.markdown(f'<div class="{badge_class}">{badge_text}</div>', unsafe_allow_html=True)
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_prob * 100,
                    title={'text': "Risk Probability", 'font': {'color': '#e0e0ff'}},
                    number={'suffix': '%', 'font': {'color': '#e0e0ff'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#a0a0c0'},
                        'bar': {'color': '#ff6b6b' if is_risk else '#2ecc71'},
                        'bgcolor': 'rgba(255,255,255,0.05)',
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(46,204,113,0.2)'},
                            {'range': [30, 70], 'color': 'rgba(243,156,18,0.2)'},
                            {'range': [70, 100], 'color': 'rgba(255,107,107,0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': 'white', 'width': 2},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_areas:
                st.markdown("#### Identified Weak Areas")
                for area in gap_result['weak_areas']:
                    sev = area['severity']
                    color = {'high': '#ff6b6b', 'medium': '#f39c12', 'low': '#2ecc71'}[sev]
                    icon = {'high': '&#9888;', 'medium': '&#9888;', 'low': '&#8505;'}[sev]
                    st.markdown(f"""
                    <div class="resource-card weak-area-{sev}" style="border-left: 4px solid {color};">
                        <div style="color: {color}; font-weight: 600; font-size: 0.85rem;">{icon} {sev.upper()} SEVERITY</div>
                        <div style="color: #e0e0ff; font-weight: 500; margin-top: 4px;">{area['area']}</div>
                        <div style="color: #a0a0c0; font-size: 0.85rem; margin-top: 2px;">{area['detail']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # TAB 3: RESOURCES
        with tab3:
            st.markdown("#### Recommended Learning Resources")
            for i, rec in enumerate(recs, 1):
                type_icon = {'video': '&#127909;', 'pdf': '&#128196;', 'practice': '&#9998;'}.get(rec['type'], '&#128218;')
                diff_label = {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced'}.get(rec.get('difficulty', 1), 'General')
                
                st.markdown(f"""
                <div class="resource-card">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="font-size: 1.5rem;">{type_icon}</div>
                        <div style="flex: 1;">
                            <div style="color: #e0e0ff; font-weight: 600;">{rec['name']}</div>
                            <div style="color: #a0a0c0; font-size: 0.85rem; margin-top: 2px;">
                                {rec['type'].title()} &middot; {diff_label} &middot; Match Score: {rec.get('score', 0):.2f}
                            </div>
                        </div>
                        <div style="color: #667eea; font-weight: 600;">#{i}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # TAB 4: STUDY PLAN
        with tab4:
            st.markdown(f"#### Personalized 7-Day Study Plan")
            
            intensity_color = {'intensive': '#ff6b6b', 'moderate': '#f39c12', 'light': '#2ecc71'}
            col_int, col_hrs, col_total = st.columns(3)
            with col_int:
                st.metric("Intensity", plan['intensity'].upper())
            with col_hrs:
                st.metric("Daily Hours", f"{plan['daily_hours']}h")
            with col_total:
                st.metric("Weekly Total", f"{plan['total_hours']:.0f}h")
            
            st.markdown("")
            
            for day_info in plan['days']:
                day_color = '#667eea'
                if day_info['day_number'] <= 2:
                    day_color = '#ff6b6b'
                elif day_info['day_number'] <= 4:
                    day_color = '#f39c12'
                elif day_info['day_number'] == 6:
                    day_color = '#764ba2'
                
                with st.expander(f"{day_info['day']} (Day {day_info['day_number']}) -- {day_info.get('focus', 'Study')}"):
                    st.markdown(f"**Duration:** {day_info['hours']}h")
                    st.markdown(f"**Focus:** {day_info.get('focus', 'General Study')}")
                    st.markdown("**Activities:**")
                    for activity in day_info.get('activities', []):
                        st.markdown(f"- {activity}")
                    if 'mastery_tip' in day_info:
                        st.info(f"Tip: {day_info['mastery_tip']}")
        
        # TAB 5: PROGRESS REPORT
        with tab5:
            st.markdown("#### Weekly Progress Report")
            
            # Overview cards
            g = report['grades']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                delta = f"{g['grade_change']:+d}" if g['grade_change'] != 0 else "0"
                st.metric("Final Grade", f"{g['G3']}/20", delta=delta)
            with col2:
                st.metric("Trend", g['trend'])
            with col3:
                st.metric("Overall Mastery", f"{report['mastery']['overall']}%")
            with col4:
                risk_pct = report['risk_assessment']['risk_probability']
                st.metric("Risk Level", f"{risk_pct}%")
            
            # Mastery details
            st.markdown("##### Mastery by Concept")
            mastery_df = pd.DataFrame([
                {'Concept': k.title(), 'Mastery (%)': v * 100}
                for k, v in report['mastery']['scores'].items()
            ])
            fig = px.bar(mastery_df, x='Concept', y='Mastery (%)',
                        color='Mastery (%)',
                        color_continuous_scale=['#ff6b6b', '#f39c12', '#2ecc71'],
                        range_color=[0, 100])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color='#a0a0c0', gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(color='#a0a0c0', gridcolor='rgba(255,255,255,0.05)', range=[0, 100]),
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly Goals
            st.markdown("##### Weekly Goals")
            for i, goal in enumerate(report.get('weekly_goals', []), 1):
                st.markdown(f"{i}. {goal}")
            
            # Study Plan Summary
            sp = report.get('study_plan_summary', {})
            st.markdown(f"##### Study Plan: **{sp.get('intensity', 'moderate').upper()}** "
                       f"({sp.get('daily_hours', 2)}h/day, {sp.get('total_hours', 14)}h/week)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a0a0c0; font-size: 0.85rem; padding: 1rem;">
        AI Personalized Learning Agent &middot; Powered by DKT (LSTM) + XGBoost + Collaborative Filtering
        <br/>Built with PyTorch, Scikit-learn & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
