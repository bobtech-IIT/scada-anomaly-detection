export interface DocumentChunk {
  id: string;
  docName: string;
  text: string;
  vector?: number[];
  tfidf?: Record<string, number>;
}

export class ClientVectorStore {
  private chunks: DocumentChunk[] = [];
  private stopwords: Set<string> = new Set([
    "the", "and", "a", "of", "to", "in", "is", "that", "it", "on", "for", "with", 
    "as", "was", "at", "by", "an", "be", "this", "are", "from", "or", "had", "have"
  ]);

  public clear(): void {
    this.chunks = [];
  }

  public getChunksCount(): number {
    return this.chunks.length;
  }

  /**
   * Simple character-level text chunker with overlap.
   */
  public chunkText(text: string, docName: string, chunkSize = 800, chunkOverlap = 150): void {
    if (!text || text.trim().length === 0) return;

    let startIndex = 0;
    let count = 0;

    while (startIndex < text.length) {
      let endIndex = startIndex + chunkSize;
      if (endIndex > text.length) endIndex = text.length;

      // Try to break at a space or newline to avoid cutting words
      if (endIndex < text.length) {
        const lastSpace = text.lastIndexOf(" ", endIndex);
        if (lastSpace > startIndex + chunkSize / 2) {
          endIndex = lastSpace;
        }
      }

      const chunkText = text.substring(startIndex, endIndex).trim();
      if (chunkText.length > 50) {
        const id = `${docName}_chunk_${count++}`;
        const chunk: DocumentChunk = {
          id,
          docName,
          text: chunkText,
        };
        
        // Compute local TF-IDF weights for fallback matching
        chunk.tfidf = this.computeTF(chunkText);
        this.chunks.push(chunk);
      }

      startIndex = endIndex - chunkOverlap;
      if (startIndex < 0) startIndex = 0;
      if (endIndex === text.length) break;
    }

    // Recalculate global weights for TF-IDF
    this.computeIDF();
  }

  /**
   * Computes Term Frequency for a single text chunk.
   */
  private computeTF(text: string): Record<string, number> {
    const terms = text.toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/);
    
    const tf: Record<string, number> = {};
    let totalWords = 0;

    for (const term of terms) {
      if (term.length < 2 || this.stopwords.has(term)) continue;
      tf[term] = (tf[term] || 0) + 1;
      totalWords++;
    }

    // Normalize TF
    if (totalWords > 0) {
      for (const term in tf) {
        tf[term] = tf[term] / totalWords;
      }
    }
    return tf;
  }

  /**
   * Computes Inverse Document Frequency and final TF-IDF scores for all chunks.
   */
  private computeIDF(): void {
    const N = this.chunks.length;
    if (N === 0) return;

    const df: Record<string, number> = {};
    for (const chunk of this.chunks) {
      if (!chunk.tfidf) continue;
      for (const term in chunk.tfidf) {
        df[term] = (df[term] || 0) + 1;
      }
    }

    const idf: Record<string, number> = {};
    for (const term in df) {
      idf[term] = Math.log(1 + N / df[term]);
    }

    // Multiply TF by IDF to get final weight vectors
    for (const chunk of this.chunks) {
      if (!chunk.tfidf) continue;
      for (const term in chunk.tfidf) {
        chunk.tfidf[term] = chunk.tfidf[term] * (idf[term] || 0);
      }
    }
  }

  /**
   * Client-side cosine similarity calculation for TF-IDF search.
   */
  public searchLocal(query: string, topK = 4): DocumentChunk[] {
    const queryTF = this.computeTF(query);
    if (this.chunks.length === 0) return [];

    const scores = this.chunks.map((chunk) => {
      let dotProduct = 0;
      let queryNormSq = 0;
      let chunkNormSq = 0;

      // Compute query norm
      for (const term in queryTF) {
        queryNormSq += queryTF[term] * queryTF[term];
      }

      // Compute dot product and chunk norm
      if (chunk.tfidf) {
        for (const term in chunk.tfidf) {
          chunkNormSq += chunk.tfidf[term] * chunk.tfidf[term];
          if (queryTF[term]) {
            dotProduct += queryTF[term] * chunk.tfidf[term];
          }
        }
      }

      const queryNorm = Math.sqrt(queryNormSq);
      const chunkNorm = Math.sqrt(chunkNormSq);
      const similarity = queryNorm > 0 && chunkNorm > 0 
        ? dotProduct / (queryNorm * chunkNorm)
        : 0;

      return { chunk, score: similarity };
    });

    // Sort by score descending and return top K
    return scores
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((s) => s.chunk);
  }

  /**
   * Set API embeddings if they are fetched from external embedding providers.
   */
  public addChunkVector(chunkId: string, vector: number[]): void {
    const chunk = this.chunks.find((c) => c.id === chunkId);
    if (chunk) {
      chunk.vector = vector;
    }
  }

  /**
   * Search vector embeddings using cosine similarity.
   */
  public searchVectors(queryVector: number[], topK = 4): DocumentChunk[] {
    if (this.chunks.length === 0 || !this.chunks[0].vector) {
      return [];
    }

    const scores = this.chunks.map((chunk) => {
      const vec = chunk.vector;
      if (!vec) return { chunk, score: 0 };

      let dotProduct = 0;
      let normA = 0;
      let normB = 0;

      for (let i = 0; i < vec.length; i++) {
        dotProduct += queryVector[i] * vec[i];
        normA += queryVector[i] * queryVector[i];
        normB += vec[i] * vec[i];
      }

      const similarity = normA > 0 && normB > 0
        ? dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
        : 0;

      return { chunk, score: similarity };
    });

    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((s) => s.chunk);
  }
}
