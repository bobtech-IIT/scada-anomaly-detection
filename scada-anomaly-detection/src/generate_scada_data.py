import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "scada_telemetry.csv")

def generate_synthetic_data(num_samples=5000):
    """Generates synthetic time-series SCADA data with injected anomalies."""
    print("Generating synthetic SCADA telemetry...")
    
    # Base timestamp
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=10 * i) for i in range(num_samples)]
    
    # Normal operation distributions
    np.random.seed(42)
    rpm = np.random.normal(loc=15.0, scale=1.5, size=num_samples)
    temperature = np.random.normal(loc=65.0, scale=5.0, size=num_samples)
    vibration = np.random.normal(loc=1.2, scale=0.3, size=num_samples)
    
    # Inject Anomalies (approx 2% of data)
    num_anomalies = int(num_samples * 0.02)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Simulate an overheating bearing or gearbox
        temperature[idx] += np.random.uniform(20.0, 35.0)
        vibration[idx] += np.random.uniform(2.0, 4.0)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "turbine_id": ["T-402"] * num_samples,
        "rotor_rpm": rpm,
        "gearbox_temp_c": temperature,
        "vibration_mm_s": vibration
    })
    
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset generated successfully at: {OUTPUT_FILE}")
    print(f"Total records: {num_samples} | Injected Anomalies: {num_anomalies}")

if __name__ == "__main__":
    generate_synthetic_data()
