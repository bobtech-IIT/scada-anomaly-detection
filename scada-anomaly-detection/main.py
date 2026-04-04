import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from generate_scada_data import generate_synthetic_data
from anomaly_model import SCADAAnomalyDetector
from alert_system import IncidentAlertSystem

DATA_PATH = "data/scada_telemetry.csv"

def main():
    print("=====================================================")
    print(" SCADA-Driven Anomaly Detection & Reporting Pipeline ")
    print("=====================================================")
    
    # 1. Generate Data if missing
    if not os.path.exists(DATA_PATH):
        generate_synthetic_data()
        
    # 2. Train Model
    detector = SCADAAnomalyDetector(contamination=0.02)
    detector.train(DATA_PATH)
    
    # 3. Simulate streaming/batch inference
    print("\nRunning simulated batch inference...")
    df = pd.read_csv(DATA_PATH)
    
    # Just grab a random sample of 200 rows to simulate streaming check
    sample_df = df.sample(n=200, random_state=1)
    
    results = detector.detect(sample_df)
    
    anomalies = results[results['is_anomaly']]
    
    print(f"\nInference Complete. Checked {len(sample_df)} records.")
    print(f"Detected {len(anomalies)} anomalies requiring attention.")
    
    # 4. Generate Reports for anomalous events
    alert_sys = IncidentAlertSystem()
    
    if len(anomalies) > 0:
        print("\nTriggering Incident Automated Reporting...")
        for _, row in anomalies.iterrows():
            report = alert_sys.generate_report(
                turbine_id=row['turbine_id'],
                timestamp=row['timestamp'],
                temp=row['gearbox_temp_c'],
                vib=row['vibration_mm_s'],
                rpm=row['rotor_rpm']
            )
            alert_sys.route_alert(report)
            break # Just show the first one for demonstration

if __name__ == "__main__":
    main()
