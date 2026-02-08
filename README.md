
# AgriFlow: Autonomous Logistics & Alpha Signal Platform

**AgriFlow** is a dual-engine architecture designed to solve the two biggest problems in agriculture: **Physical Waste** (Logistics) and **Financial Risk** (Market Volatility).

---

## 🏗️ System Architecture

The system is composed of two interacting neural engines:
1.  **AgriFlow Physics Engine (The "Twin"):** A Deep Reinforcement Learning (DRL) agent that simulates and optimizes physical fleet logistics in real-time.
2.  **AgriAlpha Signal Engine (The "Predator"):** A Gradient Boosting Machine that ingests satellite data + the *output* of the Physics Engine to predict commodity price movements.

### 🧠 Engine 1: The Logistics RL Agent (Detailed Mechanics)
The core of AgriFlow is a **Policy Gradient Agent (REINFORCE Algorithm)** trained to master the *Stochastic Vehicle Routing Problem (SVRP)*.

#### **1. The State Space (Input)**
The Agent observes a normalized vector ($S_t$) of dimension ~654 at every timestep:
*   **Fleet Context (95 dims):** Location of every truck (Lat/Lon embedded).
*   **Inventory Levels (180 dims):** Current milk volume at every farm (% full).
*   **Demand Signals (368 dims):** Backlog at every processing plant.
*   **Environmental Tensors:**
    *   `Traffic Entropy`: A calculated measure of network congestion derived from Snowfall/Rain.

### 📈 Engine 2: The Alpha Predictor (Gradient Boosting V2)
The Alpha Engine does not just look at price history. It looks at **Physical Causality**.

#### **1. The Proprietary "Logistics Breakage" Layer**
Unlike standard trading bots, AgriFlow runs a **Digital Twin Simulation** for every single prediction date.
*   **Step 1:** Ingest Satellite Weather Data (Snow, Ice, Rain).
*   **Step 2:** Feed this weather into the *Logistics RL Engine*.
*   **Step 3:** The Engine simulates 50 steps of truck movement to see if the supply chain *breaks*.
*   **Step 4:** It calculates a **"Logistics Disruption Score" (0-100)** based on how much the efficiency drops compared to a clear day.
*   **Result:** This score detects *physical* supply shocks (e.g., milk dumping due to road closures) *before* they hit the financial markets.

#### **2. Multi-Source Ingestion**
It aggregates 15+ distinct data streams:
*   **Proprietary:** Logistics Disruption Score (The "Alpha").
*   **Satellite:** Soil Moisture, Vegetation Index (VPD), Cloud Cover (Visibility).
*   **Macro:** Crude Oil (WTI), 10Y Treasury Yields.
*   **Technicals:** RSI (14-day), SMA (10-day, 30-day).

#### **3. The Prediction Model (XGBoost Logic)**
We use a **Gradient Boosting Regressor** (sklearn implementation) which outperforms Random Forests on sequential financial data.
*   **Features:** `[Logistics_Breakage, RSI_14, SMA_Cross, Oil_Cost, Soil_Moisture]`
*   **Target:** Day $T$ Closing Price.

---

## 🚀 Running the System
1.  **Start:** `python app_flask.py`
2.  **Dashboard:** Open `http://localhost:5000`
3.  **Modes:**
    *   **Logistics Twin:** Run the RL simulation to see "Cost Reduction".
    *   **Alpha Predator:** Use the "Time Machine". If you select a Weekend, the system automatically runs the simulation for the *next trading day* using accumulated weekend weather stress.
    *   **System Internals:** animated visualization of the data pipeline.

---
**Tech Stack:** Python 3.9, PyTorch (RL), GradientBoosting (Alpha), Flask (API), TailwindCSS (UI), Open-Meteo (Satellite), YFinance.
