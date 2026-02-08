
# AgriFlow: Autonomous Logistics & Alpha Signal Platform

**AgriFlow** is a dual-engine architecture designed to solve the two biggest problems in agriculture: **Physical Waste** (Logistics) and **Financial Risk** (Market Volatility).

---

## 🏗️ System Architecture

The system is composed of two interacting neural engines:
1.  **AgriFlow Physics Engine (The "Twin"):** A Deep Reinforcement Learning (DRL) agent that simulates and optimizes physical fleet logistics in real-time.
2.  **AgriAlpha Signal Engine (The "Predator"):** A Predictive Model that ingests satellite/weather data + the *output* of the Physics Engine to predict commodity price movements.

### 🧠 Engine 1: The Logistics RL Agent (Detailed Mechanics)
The core of AgriFlow is a **Policy Gradient Agent (REINFORCE Algorithm)** trained to master the *Stochastic Vehicle Routing Problem (SVRP)*.

#### **1. The State Space (Input)**
The Agent observes a normalized vector ($S_t$) of dimension ~654 at every timestep:
*   **Fleet Context (95 dims):** Location of every truck (Lat/Lon embedded).
*   **Inventory Levels (180 dims):** Current milk volume at every farm (% full).
*   **Demand Signals (368 dims):** Backlog at every processing plant.
*   **Environmental Tensors (11 dims):**
    *   `Traffic Entropy`: A calculated measure of network congestion derived from Snowfall/Rain data.
    *   `Thermal Stress`: Current temperature (affects spoilage rate).

#### **2. The Neural Policy (The "Brain")**
This state is fed into a **PyTorch Multi-Layer Perceptron (MLP)**:
*   `Layer 1`: 654 -> 512 neurons (ELU Activation + Dropout for robustness).
*   `Layer 2`: 512 -> 256 neurons (Feature Extraction).
*   `Head A (Source)`: Outputs probability distribution over 180 Farms.
*   `Head B (Destination)`: Outputs probability distribution over 368 Customers.

#### **3. Auto-Regressive Action Masking (Safety)**
To prevent "hallucinations" (assigning routes to empty farms):
*   **Source Mask:** We explicitly zero out logits for Farms with $< 100$ gallons. The Agent *cannot* pick them.
*   **Destination Mask:** We zero out logits for unconnected nodes (graph topology enforcement).

#### **4. The Reward Signal (Training)**
The agent was trained over **50,000 Episodes** using this reward function:
$$ R = (\text{Revenue} \times \alpha) - (\text{Distance} \times \beta) - (\text{Spoilage} \times \gamma) $$
*   **Critical Insight:** During training, we injected "Chaos" (Random Traffic Jams). The agent learned that *taking a longer route* (low $\beta$) is better than risking a traffic jam that causes spoilage (ultra-high $\gamma$).
*   **Result:** It outperforms linear solvers (like SAP) because linear solvers optimize for Distance, whereas AgriFlow optimizes for **Risk-Adjusted Survival**.

---

## 📈 Engine 2: The Alpha Predictor (Detailed Mechanics)
The Alpha Engine does not just look at price history. It looks at **Physical Causality**.

#### **1. Multi-Source Ingestion**
It aggregates 12+ distinct data streams:
*   **Satellite:** Soil Moisture, Vegetation Index (VPD), Cloud Cover (Visibility).
*   **Weather Reanalysis:** Snow Accumulation, Wind Gusts.
*   **Macro:** Crude Oil (WTI), 10Y Treasury Yields.

#### **2. The "Traffic Chaos" Derivative**
Before predicting price, we calculate a **Derived Feature**:
*   `Traffic_Stress_Index = (Snow_3Day_Sum * 2.0) + (Rain_Intensity) + (Wind_Gust_Risk)`
*   This feature proxies the *probability of supply chain failure* in the Midwest.

#### **3. The Prediction Model (Random Forest)**
We use a **Walk-Forward Validation** approach (No look-ahead bias).
*   **Training:** On Day $T$, we train on all history $H_{0...T-1}$.
*   **Features:** `[Price_Momentum, Traffic_Stress_Index, Oil_Cost, Soil_Moisture]`
*   **Target:** Day $T$ Closing Price.

#### **Why It Works?**
Markets are efficient at pricing *known* news. They are inefficient at pricing **hyper-local physical disruptions** (e.g., "The I-80 bridge is icy"). AgriFlow detects the ice (via Sat/Weather data), predicts the logistics failure, and generates a **Short/Long Signal** before the market reacts to the shortage.

---

## 🚀 Running the System
1.  **Start:** `python app_flask.py`
2.  **Dashboard:** Open `http://localhost:5000`
3.  **Modes:**
    *   **Logistics Twin:** Run the RL simulation to see "Cost Reduction" vs. Legacy systems.
    *   **Alpha Predator:** Use the "Time Machine" to backtest predictions against historical satellite data.

---
**Tech Stack:** Python 3.9, PyTorch (RL), Scikit-Learn (Alpha), Flask (API), TailwindCSS (UI), Open-Meteo (Satellite Data).
