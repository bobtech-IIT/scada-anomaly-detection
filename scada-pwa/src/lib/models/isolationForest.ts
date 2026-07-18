interface ITreeNode {
  splitFeature?: string;
  splitValue?: number;
  left?: ITreeNode;
  right?: ITreeNode;
  size: number;
  isLeaf: boolean;
}

export class IsolationForest {
  private trees: ITreeNode[] = [];
  private numTrees: number;
  private subSampleSize: number;
  private features: string[] = [];

  constructor(numTrees = 100, subSampleSize = 256) {
    this.numTrees = numTrees;
    this.subSampleSize = subSampleSize;
  }

  // Calculate the average path length of an unsuccessful search in a Binary Search Tree (BST)
  private c(n: number): number {
    if (n <= 1) return 0;
    if (n === 2) return 1;
    const eulerGamma = 0.5772156649;
    return 2 * (Math.log(n - 1) + eulerGamma) - (2 * (n - 1)) / n;
  }

  public fit(data: any[], features: string[]): void {
    this.features = features;
    this.trees = [];
    const n = data.length;
    if (n === 0) return;

    const limit = Math.ceil(Math.log2(Math.max(this.subSampleSize, 2)));

    for (let i = 0; i < this.numTrees; i++) {
      // Draw random subsample
      const sample: any[] = [];
      const sampleSize = Math.min(this.subSampleSize, n);
      const indices = new Set<number>();
      while (indices.size < sampleSize) {
        indices.add(Math.floor(Math.random() * n));
      }
      for (const idx of indices) {
        sample.push(data[idx]);
      }

      this.trees.push(this.buildTree(sample, 0, limit));
    }
  }

  private buildTree(data: any[], currentHeight: number, limit: number): ITreeNode {
    const size = data.length;
    if (currentHeight >= limit || size <= 1) {
      return { size, isLeaf: true };
    }

    // Filter out features with no variation
    const validFeatures = this.features.filter((feature) => {
      if (size === 0) return false;
      const values = data.map((d) => d[feature]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      return max > min;
    });

    if (validFeatures.length === 0) {
      return { size, isLeaf: true };
    }

    // Pick random feature
    const splitFeature = validFeatures[Math.floor(Math.random() * validFeatures.length)];
    const values = data.map((d) => d[splitFeature]);
    const min = Math.min(...values);
    const max = Math.max(...values);

    // Pick random split point
    const splitValue = Math.random() * (max - min) + min;

    const leftData = data.filter((d) => d[splitFeature] < splitValue);
    const rightData = data.filter((d) => d[splitFeature] >= splitValue);

    return {
      splitFeature,
      splitValue,
      left: this.buildTree(leftData, currentHeight + 1, limit),
      right: this.buildTree(rightData, currentHeight + 1, limit),
      size,
      isLeaf: false,
    };
  }

  private pathLength(x: any, node: ITreeNode, currentHeight: number): number {
    if (node.isLeaf) {
      return currentHeight + this.c(node.size);
    }

    const feature = node.splitFeature!;
    const val = x[feature];

    if (val === undefined) {
      return currentHeight; // Fallback
    }

    if (val < node.splitValue!) {
      return this.pathLength(x, node.left!, currentHeight + 1);
    } else {
      return this.pathLength(x, node.right!, currentHeight + 1);
    }
  }

  public computeAnomalyScore(x: any): number {
    if (this.trees.length === 0) return 0.5;

    let pathLengthSum = 0;
    for (const tree of this.trees) {
      pathLengthSum += this.pathLength(x, tree, 0);
    }

    const avgPathLength = pathLengthSum / this.trees.length;
    const subSampleSize = Math.min(this.subSampleSize, this.trees[0].size);
    const cVal = this.c(subSampleSize);

    if (cVal === 0) return 0;
    return Math.pow(2, -avgPathLength / cVal);
  }

  public predict(data: any[], threshold = 0.6): { score: number; isAnomaly: boolean }[] {
    return data.map((x) => {
      const score = this.computeAnomalyScore(x);
      return {
        score,
        isAnomaly: score >= threshold,
      };
    });
  }
}
