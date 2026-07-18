export interface PcaProjection {
  pc1: number;
  pc2: number;
  is_anomaly: boolean;
  timestamp: string;
}

export class PCA {
  private means: number[] = [];
  private stdDevs: number[] = [];
  private eigenvectors: number[][] = []; // [pc1_vector, pc2_vector]
  private features: string[] = [];

  constructor() {}

  /**
   * Fit PCA on a dataset and return 2D projections.
   */
  public fitTransform(data: any[], features: string[]): PcaProjection[] {
    this.features = features;
    const n = data.length;
    const m = features.length;
    if (n < 2 || m < 2) return [];

    // Step 1: Standardize features (mean=0, std=1)
    this.means = Array(m).fill(0);
    this.stdDevs = Array(m).fill(0);

    // Compute means
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += data[i][features[j]] || 0;
      }
      this.means[j] = sum / n;
    }

    // Compute standard deviations
    for (let j = 0; j < m; j++) {
      let sumSqDiff = 0;
      for (let i = 0; i < n; i++) {
        const diff = (data[i][features[j]] || 0) - this.means[j];
        sumSqDiff += diff * diff;
      }
      this.stdDevs[j] = Math.sqrt(sumSqDiff / (n - 1)) || 0.0001; // Avoid divide by zero
    }

    // Standardize data
    const X_scaled: number[][] = Array.from({ length: n }, () => Array(m).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        X_scaled[i][j] = ((data[i][features[j]] || 0) - this.means[j]) / this.stdDevs[j];
      }
    }

    // Step 2: Compute Covariance Matrix (m x m)
    const cov: number[][] = Array.from({ length: m }, () => Array(m).fill(0));
    for (let j1 = 0; j1 < m; j1++) {
      for (let j2 = 0; j2 < m; j2++) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          sum += X_scaled[i][j1] * X_scaled[i][j2];
        }
        cov[j1][j2] = sum / (n - 1);
      }
    }

    // Step 3: Find top 2 eigenvectors using Power Iteration & Deflation
    this.eigenvectors = [];
    
    // First Principal Component (PC1)
    const pc1 = this.powerIteration(cov, 100);
    this.eigenvectors.push(pc1.vector);

    // Deflate the Covariance matrix to remove PC1 projection
    // Cov_deflated = Cov - eigenvalue * v * v^T
    const covDeflated: number[][] = Array.from({ length: m }, () => Array(m).fill(0));
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < m; c++) {
        covDeflated[r][c] = cov[r][c] - pc1.value * pc1.vector[r] * pc1.vector[c];
      }
    }

    // Second Principal Component (PC2)
    const pc2 = this.powerIteration(covDeflated, 100);
    this.eigenvectors.push(pc2.vector);

    // Step 4: Project standardized data onto PC1 and PC2 axes
    const projections: PcaProjection[] = [];
    for (let i = 0; i < n; i++) {
      let pc1Val = 0;
      let pc2Val = 0;
      
      for (let j = 0; j < m; j++) {
        pc1Val += X_scaled[i][j] * this.eigenvectors[0][j];
        pc2Val += X_scaled[i][j] * this.eigenvectors[1][j];
      }

      projections.push({
        pc1: Math.round(pc1Val * 1000) / 1000,
        pc2: Math.round(pc2Val * 1000) / 1000,
        is_anomaly: !!data[i].is_anomaly,
        timestamp: data[i].timestamp,
      });
    }

    return projections;
  }

  /**
   * Helper: Power Iteration algorithm to find the dominant eigenvector and eigenvalue
   */
  private powerIteration(matrix: number[][], maxIterations = 100): { vector: number[]; value: number } {
    const m = matrix.length;
    let b = Array(m).fill(1); // Start with a non-zero vector

    for (let iter = 0; iter < maxIterations; iter++) {
      const nextB = Array(m).fill(0);
      
      // Matrix-vector multiplication
      for (let r = 0; r < m; r++) {
        for (let c = 0; c < m; c++) {
          nextB[r] += matrix[r][c] * b[c];
        }
      }

      // Calculate norm
      let norm = 0;
      for (let r = 0; r < m; r++) {
        norm += nextB[r] * nextB[r];
      }
      norm = Math.sqrt(norm);

      // Avoid division by zero
      if (norm === 0) {
        break;
      }

      // Re-normalize
      for (let r = 0; r < m; r++) {
        b[r] = nextB[r] / norm;
      }
    }

    // Rayleigh quotient: eigenvalue = (b^T * Matrix * b) / (b^T * b)
    // Since b is normalized, b^T * b = 1, so eigenvalue = b^T * Matrix * b
    const Ab = Array(m).fill(0);
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < m; c++) {
        Ab[r] += matrix[r][c] * b[c];
      }
    }

    let eigenvalue = 0;
    for (let r = 0; r < m; r++) {
      eigenvalue += b[r] * Ab[r];
    }

    return {
      vector: b,
      value: eigenvalue,
    };
  }

  /**
   * Get the feature loadings for PC1 and PC2 (importance of each variable)
   */
  public getLoadings(): Record<string, number[]> {
    const loadings: Record<string, number[]> = {};
    for (let j = 0; j < this.features.length; j++) {
      loadings[this.features[j]] = [
        Math.round((this.eigenvectors[0]?.[j] || 0) * 100) / 100,
        Math.round((this.eigenvectors[1]?.[j] || 0) * 100) / 100,
      ];
    }
    return loadings;
  }
}
