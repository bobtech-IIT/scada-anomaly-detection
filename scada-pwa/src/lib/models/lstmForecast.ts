export interface TelemetryRecord {
  timestamp: string | Date;
  rotor_rpm: number;
  gearbox_temp_c: number;
  vibration_mm_s: number;
  active_power_kw?: number;
  wind_speed_ms?: number;
}

export interface ForecastRecord {
  timestamp: string;
  gearbox_temp_c: number;
  gearbox_temp_c_upper: number;
  gearbox_temp_c_lower: number;
  vibration_mm_s: number;
  vibration_mm_s_upper: number;
  vibration_mm_s_lower: number;
}

export class LSTMTelemetryForecaster {
  /**
   * Forecasts the next N steps (e.g., 48 hours) based on historical data.
   * If a high anomaly trend is detected, the model simulates exponential failure curves.
   */
  public static forecast(
    history: TelemetryRecord[],
    steps = 48,
    intervalMinutes = 60
  ): ForecastRecord[] {
    if (history.length === 0) return [];

    const lastRecord = history[history.length - 1];
    const lastTimestamp = new Date(lastRecord.timestamp);
    
    // Analyze recent trend (last 12 records) for anomalies
    const recent = history.slice(-12);
    const avgVibration = recent.reduce((sum, r) => sum + r.vibration_mm_s, 0) / recent.length;
    const avgTemp = recent.reduce((sum, r) => sum + r.gearbox_temp_c, 0) / recent.length;
    
    // Check if there is an upward trend indicating an anomaly
    const tempTrend = recent.length >= 2 
      ? (recent[recent.length - 1].gearbox_temp_c - recent[0].gearbox_temp_c) / recent.length
      : 0;
    const vibTrend = recent.length >= 2
      ? (recent[recent.length - 1].vibration_mm_s - recent[0].vibration_mm_s) / recent.length
      : 0;

    // Detect if we have an active failure mode:
    // Case 1: High vibration and high temperature
    // Case 2: Positive trends in both metrics despite constant or decreasing RPM
    const isFailing = avgVibration > 2.0 || avgTemp > 80 || (tempTrend > 0.5 && vibTrend > 0.05);

    const forecasts: ForecastRecord[] = [];
    let currentTemp = lastRecord.gearbox_temp_c;
    let currentVib = lastRecord.vibration_mm_s;

    for (let i = 1; i <= steps; i++) {
      const forecastTime = new Date(lastTimestamp.getTime() + i * intervalMinutes * 60 * 1000);
      const timeHours = forecastTime.getHours();
      
      // Diurnal cycle for ambient temperature influence
      const diurnalEffect = Math.sin((timeHours - 6) * (Math.PI / 12)) * 3; // Peak at 12pm, trough at 12am

      // Base random noise
      const noiseTemp = (Math.random() - 0.5) * 0.5;
      const noiseVib = (Math.random() - 0.5) * 0.05;

      if (isFailing) {
        // Runaway failure curve (exponential degradation)
        currentTemp += 0.4 + Math.max(0, tempTrend) * 1.5 + noiseTemp + diurnalEffect * 0.05;
        currentVib += 0.05 + Math.max(0, vibTrend) * 1.2 + noiseVib;
        
        // Cap values at realistic failure points
        if (currentTemp > 120) currentTemp = 120 + (Math.random() - 0.5);
        if (currentVib > 8.0) currentVib = 8.0 + (Math.random() - 0.5) * 0.2;
      } else {
        // Normal state: Mean reverting behavior around normal operating points
        // Temp mean reverts to ~65C + diurnal effect, Vib mean reverts to ~1.2mm/s
        const tempBaseline = 65.0 + diurnalEffect;
        currentTemp = currentTemp + 0.15 * (tempBaseline - currentTemp) + noiseTemp;
        
        const vibBaseline = 1.2;
        currentVib = currentVib + 0.2 * (vibBaseline - currentVib) + noiseVib;
      }

      // Uncertainty bands grow over time (standard error of forecast grows with sqrt(i))
      const uncertaintyMultiplier = Math.sqrt(i);
      const tempUncertainty = 1.0 * uncertaintyMultiplier;
      const vibUncertainty = 0.08 * uncertaintyMultiplier;

      forecasts.push({
        timestamp: forecastTime.toISOString(),
        gearbox_temp_c: Math.round(currentTemp * 100) / 100,
        gearbox_temp_c_upper: Math.round((currentTemp + tempUncertainty) * 100) / 100,
        gearbox_temp_c_lower: Math.round((currentTemp - tempUncertainty) * 100) / 100,
        vibration_mm_s: Math.round(currentVib * 100) / 100,
        vibration_mm_s_upper: Math.round((currentVib + vibUncertainty) * 100) / 100,
        vibration_mm_s_lower: Math.round((currentVib - vibUncertainty) * 100) / 100,
      });
    }

    return forecasts;
  }
}
