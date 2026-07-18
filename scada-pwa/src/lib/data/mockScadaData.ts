import { TelemetryRecord } from '../models/lstmForecast';

export interface ScadaRecord extends TelemetryRecord {
  turbine_id: string;
  wind_speed_ms: number;
  active_power_kw: number;
  theoretical_power_kw: number;
  anomaly_score?: number;
  is_anomaly: boolean;
  anomaly_type?: string;
}

// Simple Seeded Random Number Generator for reproducible datasets
class SeededRandom {
  private seed: number;

  constructor(seed = 42) {
    this.seed = seed;
  }

  // Returns a value between 0 and 1
  public next(): number {
    const x = Math.sin(this.seed++) * 10000;
    return x - Math.floor(x);
  }

  // Returns a value between min and max
  public range(min: number, max: number): number {
    return this.next() * (max - min) + min;
  }

  // Returns a value from normal distribution
  public normal(mean: number, stdDev: number): number {
    // Box-Muller transform
    const u1 = this.next() || 0.0001; // Avoid 0
    const u2 = this.next();
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    return mean + stdDev * randStdNormal;
  }
}

/**
 * Standard power curve for a 2.0 MW wind turbine
 */
function calculatePowerCurve(windSpeed: number): number {
  const cutInSpeed = 3.0;
  const ratedSpeed = 12.0;
  const cutOutSpeed = 25.0;
  const ratedPower = 2000.0; // 2000 kW

  if (windSpeed < cutInSpeed || windSpeed > cutOutSpeed) {
    return 0;
  }
  if (windSpeed >= ratedSpeed) {
    return ratedPower;
  }
  
  // Cubic power increase between cut-in and rated speeds
  const fraction = (windSpeed - cutInSpeed) / (ratedSpeed - cutInSpeed);
  return ratedPower * Math.pow(fraction, 3);
}

export function generateScadaDataset(numSamples = 10000): ScadaRecord[] {
  const rand = new SeededRandom(12345);
  const data: ScadaRecord[] = [];
  
  // Starting timestamp: 70 days ago
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - 70);
  
  let currentTemp = 60.0;
  let currentVib = 1.0;
  let windSpeed = 8.0; // Initial

  // Define anomaly windows
  const anomalyWindows = [
    { start: 1200, end: 1280, type: "Gearbox Bearing Overheating" },
    { start: 4500, end: 4620, type: "Severe Rotor Mass Imbalance" },
    { start: 8100, end: 8180, type: "Main Shaft Bearings Wear" }
  ];

  for (let i = 0; i < numSamples; i++) {
    const timestamp = new Date(startDate.getTime() + i * 10 * 60 * 1000); // 10-minute intervals
    
    // 1. Wind Speed model: Random walk with drift back to mean (Ornstein-Uhlenbeck style)
    const baseWind = 7.5;
    const diurnalWind = Math.sin((timestamp.getHours() - 18) * (Math.PI / 12)) * 1.5; // Windier in evening
    windSpeed = windSpeed + 0.1 * (baseWind + diurnalWind - windSpeed) + rand.normal(0, 0.6);
    windSpeed = Math.max(0.5, Math.min(27.0, windSpeed));

    // 2. Active Power and RPM
    const theoreticalPower = calculatePowerCurve(windSpeed);
    let activePower = theoreticalPower;
    
    // Introduce electrical loss and turbulence noise
    if (activePower > 0) {
      activePower = activePower * rand.range(0.92, 1.02);
    }
    activePower = Math.round(Math.max(0, activePower) * 10) / 10;

    let rpm = 0;
    if (windSpeed >= 3.0) {
      rpm = 8.0 + ((windSpeed - 3.0) / (12.0 - 3.0)) * 7.5; // 8 to 15.5 RPM
      if (windSpeed >= 12.0) {
        rpm = 15.5 + rand.normal(0, 0.1); // Governor cap
      }
    }
    rpm = Math.round(Math.max(0, rpm) * 100) / 100;

    // 3. Normal physical modeling for temperature & vibration
    const timeHours = timestamp.getHours();
    const diurnalEffect = Math.sin((timeHours - 6) * (Math.PI / 12)) * 3; // Diurnal ambient swing
    
    const targetTemp = 50.0 + (activePower / 2000.0) * 18.0 + diurnalEffect;
    currentTemp = currentTemp + 0.05 * (targetTemp - currentTemp) + rand.normal(0, 0.2);
    
    const targetVib = 0.8 + (rpm / 16.0) * 0.5;
    currentVib = currentVib + 0.1 * (targetVib - currentVib) + rand.normal(0, 0.05);

    // 4. Inject Anomaly patterns
    let isAnomaly = false;
    let anomalyType = undefined;

    const activeAnomaly = anomalyWindows.find(w => i >= w.start && i <= w.end);
    if (activeAnomaly) {
      isAnomaly = true;
      anomalyType = activeAnomaly.type;
      
      if (activeAnomaly.type === "Gearbox Bearing Overheating") {
        // Temperature spikes exponentially, vibration rises moderately
        const progression = (i - activeAnomaly.start) / (activeAnomaly.end - activeAnomaly.start);
        currentTemp += 0.8 * progression + rand.range(0.2, 0.5);
        currentVib += 0.02 * progression;
      } else if (activeAnomaly.type === "Severe Rotor Mass Imbalance") {
        // Vibration spikes dramatically, temperature rises moderately
        const progression = (i - activeAnomaly.start) / (activeAnomaly.end - activeAnomaly.start);
        currentVib += 0.08 * progression + rand.range(0.1, 0.3);
        currentTemp += 0.1 * progression;
      } else if (activeAnomaly.type === "Main Shaft Bearings Wear") {
        // Both temperature and vibration rise and show high fluctuations
        currentTemp += rand.range(0.3, 0.6);
        currentVib += rand.range(0.05, 0.15);
      }
    }

    data.push({
      timestamp: timestamp.toISOString(),
      turbine_id: "T-402",
      wind_speed_ms: Math.round(windSpeed * 100) / 100,
      active_power_kw: activePower,
      theoretical_power_kw: Math.round(theoreticalPower * 10) / 10,
      rotor_rpm: rpm,
      gearbox_temp_c: Math.round(currentTemp * 100) / 100,
      vibration_mm_s: Math.round(currentVib * 100) / 100,
      is_anomaly: isAnomaly,
      anomaly_type: anomalyType
    });
  }

  return data;
}
