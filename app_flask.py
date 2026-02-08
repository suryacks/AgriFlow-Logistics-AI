
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import time
from src.logistics.simulation_engine import SimulationEngine
from src.alpha.plot import AlphaPlotter # New Import

app = Flask(__name__)
engine = SimulationEngine()
alpha_plotter = AlphaPlotter() # Initialize Alpha

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    fleet_size = int(data.get('fleet_size', 50))
    traffic_prob = float(data.get('traffic_prob', 0.1))
    heat_factor = float(data.get('heat_factor', 0.5))
    steps = int(data.get('steps', 500))
    
    # Run the Digital Twin
    df, p_sap, p_ai = engine.run_simulation(
        fleet_size_input=fleet_size,
        enable_traffic=True,
        heat_start=heat_factor,
        traffic_prob=traffic_prob,
        simulation_steps=steps
    )
    
    # Prepare JSON response
    response = {
        'profit_sap': float(p_sap),
        'profit_ai': float(p_ai),
        'timeline': {
            'x': df['Step'].tolist(),
            'y_sap': df['Traditional (SAP)'].tolist(),
            'y_ai': df['AgriFlow (AI)'].tolist()
        }
    }
    return jsonify(response)

@app.route('/generate_alpha_chart', methods=['POST'])
def generate_alpha():
    data = request.json
    start = data.get('start_date', '2024-01-01')
    end = data.get('end_date', '2024-03-31')
    
    buf = alpha_plotter.generate_correlation_chart(start, end)
    
    if buf:
        return send_file(buf, mimetype='image/png')
    else:
        return jsonify({"error": "No Data Found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
