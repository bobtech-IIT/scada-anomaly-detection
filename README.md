# Aegis SCADA | Advanced Predictive Maintenance & RAG PWA

A next-generation, production-ready Progressive Web Application (PWA) designed for SCADA anomaly detection, advanced sequence forecasting, reinforcement learning operations scheduling, and Retrieval-Augmented Generation (RAG) documentation search.

Aegis SCADA transforms raw telemetry into actionable board-room reports and automated technician maintenance plans, all running directly in the browser with offline capability.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fbobtech-IIT%2Fscada-anomaly-detection)

---

## Key Architectures & Data Science Core

### 1. Outlier Classification (Isolation Forest)
The system utilizes a client-side port of the **Isolation Forest** anomaly detection algorithm. Written in TypeScript, the detector computes anomaly scores (between 0.0 and 1.0) on rolling SCADA streams (monitoring Rotor RPM, Gearbox Oil Temperature, and Bearing Vibration). Any score $\ge 0.60$ flags an outlier event and initiates automatic sequence forecasting.

### 2. Time-Series Projections (LSTM Sequence Model)
To predict impending bearing seizures, the system employs an autoregressive sequence model simulating a deep **LSTM network**. When an anomaly is detected, the model projects telemetry trends 48 hours into the future, incorporating cumulative standard error uncertainty bands. If degradation patterns are found, the model projects exponential runaway heating curves, forecasting when the temperature will exceed the critical $85^\circ\text{C}$ alarm limit.

### 3. Operational Policy Optimization (Reinforcement Learning)
A tabular **Q-Learning RL Agent** determines the optimal operational decisions. It balances the trade-offs of:
*   Running at full capacity (gaining revenue depending on fluctuating grid electricity prices).
*   Performing diagnostics (running at 40% capacity, costing \$50, but resetting wear rates).
*   Triggering an urgent scheduled shutdown (generating \$0 revenue, costing \$250, but resetting turbine health to 100%).
*   Catastrophic bearing seizure (occurring if operations continue under critical wear, penalizing the environment -\$500).

The agent updates its policy in real-time using the Bellman Equation:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### 4. Custom Document Customization (Conversational RAG)
A client-side vector database performs Retrieval-Augmented Generation (RAG) on uploaded manuals, specifications, and SOPs.
*   **Chunker**: Splits documents into overlapping 800-character segments.
*   **Vector Engine**: Computes local TF-IDF matrices (offline mode) or generates true vector embeddings via OpenAI/Gemini endpoints.
*   **Similarity Search**: Utilizes cosine similarity to match query vectors against document chunks, appending context directly to the LLM.

### 5. BYOK AI Brain (Bring Your Own Key)
By default, Aegis SCADA falls back to the **OpenRouter Free API** using `google/gemma-2-9b-it:free` as the main analysis brain. The Settings Panel allows users to securely paste custom API keys (saved strictly to browser `localStorage` for privacy) to leverage other models:
*   **OpenAI**: `gpt-4o-mini`, `gpt-4o`
*   **Google Gemini**: `gemini-1.5-flash`, `gemini-1.5-pro`
*   **Anthropic**: `claude-3-5-sonnet-20240620`

---

## Project Structure

*   `scada-pwa/`: The production Next.js PWA codebase.
    *   `src/lib/models/`: TypeScript implementations of the ML core (`isolationForest.ts`, `lstmForecast.ts`, `rlAgent.ts`).
    *   `src/lib/rag/`: Local vector search and parser engine (`vectorStore.ts`).
    *   `src/lib/api/`: Unified LLM connectors (`llm.ts`, `/api/chat/route.ts`).
    *   `src/app/`: Next.js frontend pages, layouts, and styles.
    *   `public/`: PWA service workers (`sw.js`) and web manifests.
*   `scada-anomaly-detection/`: Legacy Python CLI anomaly detection pipeline (refer to it for local data science experiments).

---

## Quick Start & Local Execution

### Prerequisites
*   [Node.js 18+](https://nodejs.org/)

### Installation
1. Navigate to the PWA folder:
    ```bash
    cd scada-pwa
    ```
2. Install dependencies:
    ```bash
    npm install
    ```
3. Run the development server:
    ```bash
    npm run dev
    ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Vercel Deployment

Deploy the SCADA PWA directly to Vercel in seconds:

1. Push your changes to your GitHub repository.
2. Link your repository in the [Vercel Dashboard](https://vercel.com).
3. In the Build & Development Settings:
    *   **Root Directory**: Set to `scada-pwa`.
    *   **Framework Preset**: Select **Next.js**.
4. Click **Deploy**. Vercel will build the React pages and spin up the backend proxy routes as Serverless Functions automatically.
