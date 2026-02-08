
# AgriFlow: Autonomous Logistics Engine

AgriFlow is a Deep Reinforcement Learning (RL) agent optimized for perishable supply chain logistics. It outperforms traditional linear optimization methods by adapting to dynamic environmental factors such as traffic congestion and spoilage risks.

## Problem Statement
Legacy logistics systems utilize static optimization models (MILP) that assume constant travel times and inventory levels. In high-variance environments involving perishable goods, these assumptions fail, leading to significant revenue loss (+20% operational waste).

## Solution Architecture
AgriFlow utilizes a centralized fleet dispatcher agent based on a multi-head Neural Network architecture.
- **Input:** 600+ dimensional state vector (GPS fleet positions, real-time inventory levels, traffic sensor data).
- **Core Logic:** Deep RL with Auto-Regressive Action Masking to ensure strict topological validity.
- **Optimization Objective:** Maximize net profit (Revenue - Cost - Spoilage Penalty).

## Performance Benchmarks
Simulations conducted on a regional dairy distribution network (95 trucks, 180 farms, 548 customers) demonstrate:
1. **Efficiency:** 20% reduction in deadhead miles through intelligent inventory balancing.
2. **Resilience:** 74% reduction in losses during severe traffic disruption events compared to static baselines.

## Deployment
### Prerequisites
- Python 3.8+
- PyTorch
- NumPy / Pandas

### Installation
```bash
pip install -r requirements.txt
```

### Execution
Launch the Web Dashboard to visualize the simulation:
```bash
python app_flask.py
```
Access the dashboard at `http://localhost:5000`.

## License
Proprietary. All rights reserved.
