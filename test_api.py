
import requests
import json
import time

def test_simulation():
    url = "http://127.0.0.1:5000/run_simulation"
    payload = {
        "fleet_size": "50",
        "traffic_prob": "0.1",
        "heat_factor": "0.5",
        "steps": "100"
    }
    
    print(f"--- Testing Logistics Sim: {url} ---")
    try:
        start = time.time()
        response = requests.post(url, json=payload)
        duration = time.time() - start
        
        print(f"Status Code: {response.status_code}")
        print(f"Duration: {duration:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print("SUCCESS! Response Keys:", data.keys())
            print(f"AI Profit: ${data['profit_ai']}")
        else:
            print("FAILURE! Response:", response.text)
            
    except Exception as e:
        print(f"Connection Error: {e}")

def test_prediction():
    url = "http://127.0.0.1:5000/predict_event"
    payload = {
        "date": "2024-01-16",
        "ticker": "DC=F"
    }
    
    print(f"\n--- Testing Alpha Prediction: {url} ---")
    try:
        response = requests.post(url, json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print("LOGIC FAILURE:", data['error'])
            else:
                print("SUCCESS!", json.dumps(data, indent=2))
        else:
            print("FAILURE! Response:", response.text)
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_simulation()
    test_prediction()
