
import streamlit as st
import pandas as pd
import numpy as np
import time
from src.logistics.simulation_engine import SimulationEngine

# Page Config
st.set_page_config(page_title="AgriFlow Dashboard", page_icon="🚛", layout="wide")

# Title
st.title("🚛 AgriFlow: AI-Driven Logistics Optimization")
st.markdown("""
**AgriFlow** is an advanced Reinforcement Learning (RL) engine designed to replace static linear programming in perishable logistics.
It adapts to **Chaos Events** (Traffic, Heat, Supply Shocks) in real-time, preventing valid losses that traditional systems miss.
""")

# Sidebar
st.sidebar.header("Simulation Parameters")
fleet_size = st.sidebar.slider("Fleet Size (Trucks)", 5, 200, 50)
enable_traffic = st.sidebar.checkbox("Enable Traffic Disruptions", value=True)
traffic_prob = st.sidebar.slider("Traffic Jam Probability", 0.0, 0.5, 0.1, help="Probability of a major route clogging per step.")
heat_factor = st.sidebar.slider("Heat Wave Factor", 0.0, 1.0, 0.5, help="Higher heat increases spoilage risk for long routes.")
sim_steps = st.sidebar.slider("Simulation Steps", 100, 1000, 500)

# Initialize Engine
if 'engine' not in st.session_state:
    st.session_state.engine = SimulationEngine()

# Run Button
if st.sidebar.button("RUN SIMULATION", type="primary"):
    with st.spinner("Running Digital Twin Simulation... (Comparing SAP vs RL)"):
        df, profit_sap, profit_ai = st.session_state.engine.run_simulation(
            fleet_size_input=fleet_size,
            enable_traffic=enable_traffic,
            heat_start=heat_factor,
            traffic_prob=traffic_prob,
            simulation_steps=sim_steps
        )
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Traditional (SAP) Profit", f"${profit_sap:,.0f}", delta_color="off")
        col2.metric("AgriFlow (AI) Profit", f"${profit_ai:,.0f}", delta=f"{profit_ai - profit_sap:,.0f}")
        
        percent_gain = ((profit_ai - profit_sap) / abs(profit_sap)) * 100 if profit_sap != 0 else 0
        col3.metric("Efficiency Gain", f"{percent_gain:.1f}%")
        
        # Chart
        st.subheader("Cumulative Profit Over Time")
        st.line_chart(df, x="Step", y=["Traditional (SAP)", "AgriFlow (AI)"], color=["#FF4B4B", "#00CC96"])
        
        # Analysis
        st.subheader("Analysis")
        if profit_ai > profit_sap:
            st.success(f"**VICTORY:** AgriFlow outperformed the Traditional baseline by ${profit_ai - profit_sap:,.0f}.")
            st.markdown(f"""
            **Why AgriFlow Won:**
            1. **Traffic Avoidance:** The AI detected traffic signals and re-routed, while SAP followed the static 'shortest path' into jams.
            2. **Inventory Balancing:** The AI avoided 'Deadheading' to empty farms by checking picking masks.
            3. **Heat Adaptation:** With a Heat Factor of {heat_factor}, the AI favored shorter, safer routes to minimize spoilage penalties.
            """)
        else:
            st.warning("AgriFlow is training... In low-chaos scenarios, Traditional methods remain competitive. Increase chaos to see the AI shine.")

# Information Section
st.markdown("---")
st.subheader("How It Works")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🔴 Traditional (SAP/Linear)")
    st.markdown("""
    - **Method:** Mixed Integer Linear Programming (MILP) or Greedy Heuristics.
    - **Logic:** Calculates the 'Perfect Route' based on average speeds and distances.
    - **Weakness:** **Static.** If a road closes 1 hour after departure, the plan fails. Spoilage occurs.
    """)

with col_b:
    st.markdown("### 🟢 AgriFlow (Reinforcement Learning)")
    st.markdown("""
    - **Method:** Deep Neural Network (PPO/REINFORCE) with Auto-Regressive Masking.
    - **Logic:** 'Plays the Game' of logistics. Sees live sensors (Traffic, Heat, Inventory).
    - **Strength:** **Anti-Fragile.** Re-optimizes every second. Learns that 'Shortest Path' isn't always 'Highest Profit'.
    """)
