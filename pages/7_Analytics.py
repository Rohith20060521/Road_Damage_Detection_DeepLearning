import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from utils.alerts import render_sidebar_alerts
from utils.style import apply_custom_style, render_modern_card, render_top_nav

st.set_page_config(page_title="Strategic Analytics | SRIMS", page_icon="📊", layout="wide")

# Apply global styling
apply_custom_style()

# Top Navbar
render_top_nav("Analytics")

# Sidebar
render_sidebar_alerts()

# Page Header
st.markdown("""
<div class="animate-in" style="margin-bottom: 30px;">
    <h1><i class="fa-solid fa-chart-pie" style="color: var(--primary-blue);"></i> Strategic Infrastructure Analytics</h1>
    <p style="color: var(--text-slate-600);">Executive briefing and fiscal ROI modeling for municipal infrastructure maintenance.</p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists("gps_log.csv"):
    st.warning("No geospatial log data found. Please run a detection mission first.")
    st.stop()

df = pd.read_csv("gps_log.csv")

# --- DATA PRE-PROCESSING ---
active_df = df[df['status'] != 'Resolved'].copy()
resolved_df = df[df['status'] == 'Resolved'].copy()
total_dets = len(df)
pending_dets = len(active_df)

# RHI Recalculation
h = len(active_df[active_df['severity'] == 'high'])
m = len(active_df[active_df['severity'] == 'medium'])
l = len(active_df[active_df['severity'] == 'low'])
rhi = 100 - ( (h*20) + (m*8) + (l*2) ) / (total_dets if total_dets > 0 else 1) * 2
rhi = max(0, min(100, rhi))

# Analytics Hero Section
col1, col2, col3 = st.columns(3)
with col1:
    render_modern_card("Integrity Index (RHI)", f"{rhi:.1f}%", "fa-heart-pulse")
with col2:
    render_modern_card("Active Hazards", f"{pending_dets}", "fa-burst")
with col3:
    grade = "A" if rhi > 85 else "B" if rhi > 70 else "C" if rhi > 55 else "D" if rhi > 40 else "F"
    render_modern_card("Infrastructure Grade", grade, "fa-medal")

st.markdown("<br>", unsafe_allow_html=True)

# Executive Briefing
st.markdown("### <i class='fa-solid fa-robot' style='color: var(--primary-blue);'></i> AI Executive Briefing", unsafe_allow_html=True)
danger_zone = active_df.groupby('location_label').size().idxmax() if not active_df.empty else "None"

briefing_html = f"""
<div class="modern-card animate-in" style="border-left: 4px solid var(--primary-blue); background: #F0F9FF;">
    <p style="margin: 0; line-height: 1.6; color: var(--text-slate-900);">
        Current City infrastructure is operating at a <strong>Grade {grade}</strong> level. 
        Analysis indicates <strong>{h} critical fractures</strong> requiring immediate mobilization 
        within the <strong>{danger_zone}</strong> sector.
    </p>
    <p style="margin-top: 15px; font-size: 0.9rem; color: var(--text-slate-600);">
        <i class="fa-solid fa-circle-info"></i> Strategic Priority: High-confidence pothole rectification in high-density corridors.
    </p>
</div>
"""
st.markdown(briefing_html, unsafe_allow_html=True)

# Budget ROI Optimizer
st.markdown("### <i class='fa-solid fa-kaaba' style='color: var(--primary-blue);'></i> Budget ROI Optimizer", unsafe_allow_html=True)
with st.container():
    budget = st.slider("Fiscal Allocation (₹)", 10000, 500000, 150000, step=5000)

    # Cost Model
    active_df['repair_cost'] = active_df['severity'].map({'high': 8000, 'medium': 3500, 'low': 1500})
    active_df['rhi_impact'] = active_df['severity'].map({'high': 2.5, 'medium': 1.2, 'low': 0.4})
    active_df['roi_factor'] = active_df['rhi_impact'] / (active_df['repair_cost'] / 1000)

    priority_repairs = active_df.sort_values(by=['roi_factor', 'confidence'], ascending=False)
    selected = []
    spent = 0
    rhi_gain = 0

    for _, row in priority_repairs.iterrows():
        if spent + row['repair_cost'] <= budget:
            selected.append(row)
            spent += row['repair_cost']
            rhi_gain += row['rhi_impact']

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimized Tasks", len(selected), help="Number of repairs within budget")
    c2.metric("Allocated Fund", f"₹{spent:,}")
    c3.metric("Projected Gain", f"+{rhi_gain:.1f}%", delta_color="normal")

# Monsoon Degradation Forecast
st.markdown("### <i class='fa-solid fa-cloud-showers-heavy' style='color: var(--primary-blue);'></i> Environment Stress Simulation", unsafe_allow_html=True)
monsoon_intensity = st.select_slider("Simulated Monsoon Severity", options=["Stable", "Light", "Heavy", "Extreme"])
decay_rate = {"Stable": 0.0, "Light": 0.05, "Heavy": 0.15, "Extreme": 0.35}

future_rhi = rhi * (1 - decay_rate[monsoon_intensity])
if decay_rate[monsoon_intensity] > 0:
    st.markdown(f"""
    <div class="modern-card" style="border: 1px solid #FCA5A5; background: #FEF2F2;">
        <i class="fa-solid fa-triangle-exclamation" style="color: #EF4444;"></i> 
        Simulated {monsoon_intensity} monsoons will degrade Integrity Index to <strong style="color: #EF4444;">{future_rhi:.1f}%</strong> 
        if pending hazards remain unresolved.
    </div>
    """, unsafe_allow_html=True)

# Visualizations
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### <i class='fa-solid fa-chart-line' style='color: var(--primary-blue);'></i> Spatial & Severity Distribution", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_sev = px.pie(active_df, names='severity', color='severity', 
                    color_discrete_map={'high':'#EF4444', 'medium':'#3B82F6', 'low':'#10B981'},
                    template="plotly_white",
                    title="Active Hazard Severity Split")
    fig_sev.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_sev, width='stretch')

with col2:
    loc_counts = active_df.groupby('location_label').size().reset_index(name='count')
    fig_loc = px.bar(loc_counts, x='location_label', y='count', 
                    template="plotly_white",
                    title="Hazard density by sector", 
                    color='count', color_continuous_scale="Blues")
    fig_loc.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig_loc, width='stretch')
