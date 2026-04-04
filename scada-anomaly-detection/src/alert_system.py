from string import Template
from datetime import datetime

class IncidentAlertSystem:
    def __init__(self):
        self.report_template = Template(
            """
[URGENT ALARM] Predictive Maintenance Alert - Turbine $turbine_id

Timestamp: $timestamp

Anomaly Detected in Telemetry Metrics:
- Gearbox Temp: $temp °C
- Vibration: $vib mm/s
- RPM: $rpm

NLP Assessment: The combination of elevated gearbox temperatures and high vibrational frequency indicates an 85% probability of bearing failure within the next 48 hours.

Action Required: Immediate inspection of Turbine $turbine_id. Suspend automated reboot sequence.
"""
        )

    def generate_report(self, turbine_id, timestamp, temp, vib, rpm):
        """Generates a structured NLP incident report based on detected anomalies."""
        report = self.report_template.substitute(
            turbine_id=turbine_id,
            timestamp=timestamp,
            temp=round(temp, 2),
            vib=round(vib, 2),
            rpm=round(rpm, 2)
        )
        return report

    def route_alert(self, report):
        """Simulates routing the alert to regional stakeholders."""
        print("\n" + "="*50)
        print("ROUTING SILENT ALERT TO MAINTENANCE API...")
        print(report)
        print("="*50 + "\n")
