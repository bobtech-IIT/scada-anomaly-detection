import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

class SCADAAnomalyDetector:
    def __init__(self, contamination=0.03):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.features = ['rotor_rpm', 'gearbox_temp_c', 'vibration_mm_s']

    def train(self, data_path):
        """Train the model on the SCADA dataset."""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        X = df[self.features]
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training Isolation Forest Model...")
        self.model.fit(X_scaled)
        
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        print(f"Saving models to {MODEL_DIR}...")
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        print("Training complete.")

    def detect(self, df):
        """Predict anomalies on a given DataFrame."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Model not found. Please train first.")
            
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        X = df[self.features]
        X_scaled = scaler.transform(X)
        
        # Isolation forest returns -1 for anomalies, 1 for normal
        predictions = model.predict(X_scaled)
        
        df_result = df.copy()
        df_result['is_anomaly'] = predictions == -1
        return df_result
