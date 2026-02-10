
import pandas as pd
import os
import sys

# Since we are in src/logistics, we go up two levels to reach root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def inspect_data():
    files = [
        "Location Demand.XLSX",
        "Milk Availability at Route.XLSX",
        "Location Receving Capacity.XLSX",
        "Transportation lane from Route to Location.xlsx",
        "Shipping and Receiving Time.xlsx"
    ]

    print(f"--- AGRIFLOW LOGISTICS DATA INSPECTION ---")
    print(f"Searching in: {DATA_DIR}\n")

    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"[{f}]")
        try:
            # Load Data
            df = pd.read_excel(path)
            
            # Print Stats
            print(f"   Shape: {df.shape} (Rows, Cols)")
            print(f"   Columns: {list(df.columns)}")
            
            # Preview first 2 rows to see content type
            print("   Preview:")
            print(df.head(2).to_string(index=False))
            print("-" * 50)
            
        except Exception as e:
            print(f"   ERROR: Could not load file. {e}")
            print("-" * 50)

if __name__ == "__main__":
    inspect_data()
