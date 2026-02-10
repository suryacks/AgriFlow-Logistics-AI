
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import time
from src.core.simulation_engine import SimulationEngine
from src.core.l2l.wrapper import L2LInterface # New
from src.alpha.plot import AlphaPlotter 
from src.alpha.predictor import AgriAlphaPredictor # New

app = Flask(__name__)
engine = SimulationEngine()
alpha_plotter = AlphaPlotter()
predictor = AgriAlphaPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_event', methods=['POST'])
def predict_event():
    data = request.json
    date = data.get('date', '2024-01-15')
    ticker = data.get('ticker', 'DC=F')
    
    result = predictor.predict_single_event(date, ticker)
    return jsonify(result)

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.json
        print(f"DEBUG: Received Payload: {data}")
        
        # Explicit Casting with Default Fallbacks
        fleet_size = int(data.get('fleet_size', 50))
        traffic_prob = float(data.get('traffic_prob', 0.1))
        heat_factor = float(data.get('heat_factor', 0.5))
        steps = int(data.get('steps', 500))
        
        print(f"DEBUG: Running Sim with Fleet={fleet_size}, Traffic={traffic_prob}, Heat={heat_factor}")
        
        # Run the Digital Twin
        df, p_sap, p_ai = engine.run_simulation(
            fleet_size_input=fleet_size,
            enable_traffic=True,
            heat_start=heat_factor,
            traffic_prob=traffic_prob,
            simulation_steps=steps
        )
        
        # User Request: "Make costs positive... smaller is better".
        # We invert Profit to represent "Operational Deficit/Cost".
        # If Profit is negative (Loss), Cost is Positive.
        # If Profit is positive (Gain), Cost is Negative (Surplus).
        cost_sap = -float(p_sap)
        cost_ai = -float(p_ai)
        
        # Invert the timeline curves too for the chart
        df['Traditional (SAP)'] = -df['Traditional (SAP)']
        df['AgriFlow (AI)'] = -df['AgriFlow (AI)']
        
        # Prepare JSON response
        response = {
            'cost_sap': cost_sap,
            'cost_ai': cost_ai,
            'timeline': {
                'x': df['Step'].tolist(),
                'y_sap': df['Traditional (SAP)'].tolist(),
                'y_ai': df['AgriFlow (AI)'].tolist()
            }
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"CRITICAL SIM ERROR: {error_msg}")
        return jsonify({"error": str(e), "trace": error_msg}), 500


@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    data = request.json
    start = data.get('start', '2024-01-01')
    end = data.get('end', '2024-03-31')
    ticker = data.get('ticker', 'DC=F')
    capital = float(data.get('capital', 10000))
    
    predictor = AgriAlphaPredictor()
    result = predictor.run_backtest(start_date=start, end_date=end, ticker=ticker, initial_capital=capital)
    return jsonify(result)

@app.route('/generate_alpha_chart', methods=['POST'])
def generate_alpha():
    data = request.json
    start = data.get('start_date', '2024-01-01')
    end = data.get('end_date', '2024-03-31')
    comm = data.get('commodity', 'DC=F')
    
    buf = alpha_plotter.generate_correlation_chart(start, end, ticker=comm)
    
    if buf:
        return send_file(buf, mimetype='image/png')
    else:
        return jsonify({"error": "No Data Found"}), 404

@app.route('/run_l2l_demo', methods=['POST'])
def run_l2l_demo():
    try:
        data = request.json
        obstacles = data.get('obstacles', [])
        
        l2l = L2LInterface()
        # New Signature: nodes_count=40, obstacles=[]
        result = l2l.run_grid_demo(obstacles=obstacles)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
