export class MaintenanceRLAgent {
  // Q-table: 27 states (3 anomaly levels * 3 time-to-failure levels * 3 energy prices) x 3 actions
  public qTable: number[][];
  private alpha: number; // Learning rate
  private gamma: number; // Discount factor
  private epsilon: number; // Exploration rate
  private decayRate: number;

  constructor(alpha = 0.1, gamma = 0.9, epsilon = 0.3) {
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.decayRate = 0.995;
    
    // Initialize Q-table with zeros
    this.qTable = Array.from({ length: 27 }, () => Array(3).fill(0));
  }

  /**
   * Encodes the continuous state variables into a discrete state index (0 to 26).
   */
  public encodeState(anomalyScore: number, hoursToFailure: number, gridPrice: number): number {
    // 1. Anomaly Level (0: Normal < 0.45, 1: Warning 0.45 - 0.65, 2: Critical >= 0.65)
    let anomalyLevel = 0;
    if (anomalyScore >= 0.65) anomalyLevel = 2;
    else if (anomalyScore >= 0.45) anomalyLevel = 1;

    // 2. Time to Failure (0: Safe > 36h, 1: Impending 12-36h, 2: Critical < 12h)
    let failureRisk = 0;
    if (hoursToFailure <= 12) failureRisk = 2;
    else if (hoursToFailure <= 36) failureRisk = 1;

    // 3. Grid Price (0: Low < $40/MWh, 1: Medium $40-$80/MWh, 2: High >= $80/MWh)
    let priceLevel = 0;
    if (gridPrice >= 80) priceLevel = 2;
    else if (gridPrice >= 40) priceLevel = 1;

    return anomalyLevel * 9 + failureRisk * 3 + priceLevel;
  }

  /**
   * Decodes a state index back into readable labels for UI rendering.
   */
  public decodeState(state: number): { anomaly: string; risk: string; price: string } {
    const anomalyLevel = Math.floor(state / 9);
    const failureRisk = Math.floor((state % 9) / 3);
    const priceLevel = state % 3;

    const anomalies = ["Normal", "Warning", "Critical"];
    const risks = ["Safe (>36h)", "Impending (12-36h)", "Critical (<12h)"];
    const prices = ["Low (<$40)", "Medium ($40-$80)", "High (>$80)"];

    return {
      anomaly: anomalies[anomalyLevel],
      risk: risks[failureRisk],
      price: prices[priceLevel],
    };
  }

  /**
   * Action selection using Epsilon-Greedy strategy.
   * Actions: 0 = Do Nothing (Keep Running), 1 = Diagnostic Check, 2 = Schedule Maintenance
   */
  public selectAction(state: number, forceExploit = false): number {
    if (!forceExploit && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * 3); // Explore
    }
    
    // Exploit: pick action with max Q-value
    const actions = this.qTable[state];
    let maxVal = -Infinity;
    let bestAction = 0;
    
    for (let a = 0; a < 3; a++) {
      if (actions[a] > maxVal) {
        maxVal = actions[a];
        bestAction = a;
      }
    }
    return bestAction;
  }

  /**
   * Update the Q-value based on the reward and transition.
   */
  public updateQ(state: number, action: number, reward: number, nextState: number): void {
    const currentQ = this.qTable[state][action];
    const maxNextQ = Math.max(...this.qTable[nextState]);
    
    // Temporal Difference target
    const tdTarget = reward + this.gamma * maxNextQ;
    // Bellman Equation update
    this.qTable[state][action] = currentQ + this.alpha * (tdTarget - currentQ);
  }

  /**
   * Simulates the transition and calculates the reward.
   */
  public step(
    anomalyScore: number,
    hoursToFailure: number,
    gridPrice: number,
    action: number
  ): {
    nextAnomalyScore: number;
    nextHoursToFailure: number;
    nextGridPrice: number;
    reward: number;
    description: string;
    isFailed: boolean;
  } {
    let nextAnomalyScore = anomalyScore;
    let nextHoursToFailure = hoursToFailure;
    let nextGridPrice = Math.round(20 + Math.random() * 100); // Random market price fluctuation
    let reward = 0;
    let description = "";
    let isFailed = false;

    // Base variables
    const basePowerRevenue = gridPrice * 1.5; // Scale electricity price to dollars generated

    if (action === 0) {
      // Action: Do Nothing (Run turbine)
      // Check for catastrophic failure if risk is high
      const failureProb = anomalyScore > 0.7 ? 0.35 : (anomalyScore > 0.5 ? 0.08 : 0.01);
      
      if (Math.random() < failureProb || hoursToFailure <= 0) {
        // Catastrophic failure occurred!
        isFailed = true;
        reward = -500; // Heavy penalty for failure
        nextAnomalyScore = 0.95;
        nextHoursToFailure = 0;
        description = "Gearbox seized! Catastrophic failure occurred. Total shutdown forced.";
      } else {
        // Normal generation reward
        reward = basePowerRevenue;
        // Natural aging: anomaly score rises slowly, time-to-failure drops
        nextAnomalyScore = Math.min(1.0, anomalyScore + 0.01 + Math.random() * 0.02);
        nextHoursToFailure = Math.max(0, hoursToFailure - 1);
        description = `Operating normally. Generated $${Math.round(reward)} revenue.`;
      }
    } else if (action === 1) {
      // Action: Diagnostics (Minor downtime / optimization check)
      const cost = -50;
      reward = cost + basePowerRevenue * 0.4; // Running at 40% efficiency during check
      // Reduces anomaly score slightly by recalibrating / lubricating
      nextAnomalyScore = Math.max(0.1, anomalyScore - 0.1);
      nextHoursToFailure = Math.min(48, hoursToFailure + 4);
      description = "Running diagnostics and lubrication. Operating at partial power. Cost: $50.";
    } else {
      // Action: Maintenance (Full shutdown & replacement)
      const cost = -250; // Major repair cost
      reward = cost; // No power generation
      // Fully resets the turbine health
      nextAnomalyScore = 0.15;
      nextHoursToFailure = 48; // Reset time to failure
      description = "Completed scheduled gearbox maintenance. Turbine health restored. Cost: $250.";
    }

    return {
      nextAnomalyScore: Math.round(nextAnomalyScore * 100) / 100,
      nextHoursToFailure,
      nextGridPrice,
      reward,
      description,
      isFailed,
    };
  }

  /**
   * Runs offline training to simulate learning.
   */
  public train(episodes = 500): { episode: number; totalReward: number; epsilon: number }[] {
    const history: { episode: number; totalReward: number; epsilon: number }[] = [];

    for (let ep = 0; ep < episodes; ep++) {
      // Reset state at start of episode
      let anomaly = 0.15;
      let ttFailure = 48;
      let price = 50;
      let state = this.encodeState(anomaly, ttFailure, price);
      let totalReward = 0;
      let steps = 0;

      // Simulate a 72-hour operating window per episode
      while (steps < 72) {
        const action = this.selectAction(state);
        const next = this.step(anomaly, ttFailure, price, action);

        const nextState = this.encodeState(
          next.nextAnomalyScore,
          next.nextHoursToFailure,
          next.nextGridPrice
        );

        this.updateQ(state, action, next.reward, nextState);

        totalReward += next.reward;
        anomaly = next.nextAnomalyScore;
        ttFailure = next.nextHoursToFailure;
        price = next.nextGridPrice;
        state = nextState;

        steps++;
        if (next.isFailed) {
          // Force maintenance to recover
          anomaly = 0.15;
          ttFailure = 48;
          state = this.encodeState(anomaly, ttFailure, price);
        }
      }

      history.push({
        episode: ep + 1,
        totalReward: Math.round(totalReward),
        epsilon: Math.round(this.epsilon * 100) / 100,
      });

      // Decay exploration rate
      this.epsilon = Math.max(0.01, this.epsilon * this.decayRate);
    }

    return history;
  }
}
