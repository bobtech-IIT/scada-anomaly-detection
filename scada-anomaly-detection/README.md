# SCADA-Driven Anomaly Detection & Automated Alerting

## Overview
This project implements predictive machine learning models to analyze time-series SCADA data from renewable energy assets (Wind Turbines/Solar Panels). The goal is to detect equipment anomalies *prior* to critical failure, maximizing operational uptime. Furthermore, it incorporates NLP to automatically draft incident reports and route targeted alerts to maintenance stakeholders when an anomaly is detected.

## Features
- **Predictive ML Models:** Utilizes Isolation Forest (and frameworks scalable to LSTMs) to identify statistical outliers in operational telemetry (vibration, temperature, RPM).
- **Automated NLP Reporting:** Employs template-based NLP generation (or lightweight LLM generation) to auto-draft easily readable incident reports for technicians.
- **Simulated SCADA Pipeline:** Includes a synthetic data generator that creates realistic failure patterns for testing the model.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Scikit-learn, Pandas, Numpy

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
1. Generate the synthetic SCADA dataset:
   ```bash
   python src/generate_scada_data.py
   ```
2. Train the model and run the monitoring pipeline:
   ```bash
   python main.py
   ```

## Repository Structure
- `data/`: Contains synthetic `.csv` extracts from the SCADA system.
- `src/`: 
  - `generate_scada_data.py`: Mocks turbine sensor data with injected anomalies.
  - `anomaly_model.py`: Handles model training, feature scaling, and prediction scoring.
  - `alert_system.py`: Converts model output into human-readable Slack/Email style alerts.
- `main.py`: Orchestrates the ML pipeline from data ingestion to alerting.
