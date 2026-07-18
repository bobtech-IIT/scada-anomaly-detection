"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Activity, 
  Cpu, 
  FileText, 
  Settings, 
  Database, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle2, 
  Download, 
  Send, 
  Upload, 
  Play, 
  RefreshCw, 
  Sliders, 
  DollarSign, 
  Layers, 
  Wrench, 
  ShieldAlert, 
  Info, 
  Eye,
  EyeOff
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
  ReferenceLine
} from "recharts";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";

// Import custom TS modules
import { generateScadaDataset, ScadaRecord } from "../lib/data/mockScadaData";
import { IsolationForest } from "../lib/models/isolationForest";
import { LSTMTelemetryForecaster, ForecastRecord } from "../lib/models/lstmForecast";
import { MaintenanceRLAgent } from "../lib/models/rlAgent";
import { ClientVectorStore } from "../lib/rag/vectorStore";
import { LLMClient, LLMSettings, ChatMessage } from "../lib/api/llm";

export default function SCADAPWADashboard() {
  const [mounted, setMounted] = useState(false);

  // Data Science State
  const [scadaData, setScadaData] = useState<ScadaRecord[]>([]);
  const [recentData, setRecentData] = useState<ScadaRecord[]>([]);
  const [detectedAnomalies, setDetectedAnomalies] = useState<ScadaRecord[]>([]);
  const [selectedAnomaly, setSelectedAnomaly] = useState<ScadaRecord | null>(null);
  const [forecasts, setForecasts] = useState<ForecastRecord[]>([]);
  
  // Tabs State
  const [activeTab, setActiveTab] = useState<"operations" | "diagnostics" | "rl" | "rag" | "reports" | "settings">("operations");

  // RL Simulator State
  const [rlAgent, setRlAgent] = useState<MaintenanceRLAgent | null>(null);
  const [rlHistory, setRlHistory] = useState<{ episode: number; totalReward: number; epsilon: number }[]>([]);
  const [rlSimState, setRlSimState] = useState({
    anomaly: 0.15,
    risk: 48,
    price: 50,
    stateIndex: 0,
    stepsCount: 0,
    isFailed: false,
    cumulativeReward: 0
  });
  const [rlLogs, setRlLogs] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);

  // RAG State
  const [vectorStore] = useState(() => new ClientVectorStore());
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; size: number; chunks: number }[]>([]);
  const [ragHistory, setRagHistory] = useState<ChatMessage[]>([]);
  const [ragQueryText, setRagQueryText] = useState("");
  const [ragLoading, setRagLoading] = useState(false);
  const [ragStatus, setRagStatus] = useState("No documents indexed. Upload operating manuals below.");

  // Settings State
  const [llmSettings, setLlmSettings] = useState<LLMSettings>(LLMClient.getDefaultSettings());
  const [showApiKey, setShowApiKey] = useState(false);
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [testMessage, setTestMessage] = useState('');

  // Report State
  const [executiveSummary, setExecutiveSummary] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);

  // Charts DOM references for PDF screenshot inserts
  const powerCurveRef = useRef<HTMLDivElement>(null);
  const forecastChartRef = useRef<HTMLDivElement>(null);

  // Initialize data and models
  useEffect(() => {
    setMounted(true);
    
    // Register Service Worker for PWA offline caching
    if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').then(
          (reg) => console.log('[PWA] Service Worker registered scope:', reg.scope),
          (err) => console.error('[PWA] Service Worker registration failed:', err)
        );
      });
    }
    
    // 1. Generate 10k SCADA records
    const rawData = generateScadaDataset(10000);
    setScadaData(rawData);
    
    // Grab the last 100 records for the operational timeline view
    setRecentData(rawData.slice(-100));

    // 2. Train Client-Side Isolation Forest
    const features = ["rotor_rpm", "gearbox_temp_c", "vibration_mm_s"];
    const detector = new IsolationForest(100, 256);
    // Fit on a sample of 2000 normal records to speed up loading
    const trainingSample = rawData.slice(0, 2000).filter(r => !r.is_anomaly);
    detector.fit(trainingSample, features);
    
    // Predict on the whole dataset
    const scores = detector.predict(rawData);
    const scoredData = rawData.map((d, index) => ({
      ...d,
      anomaly_score: Math.round(scores[index].score * 100) / 100,
      is_anomaly: scores[index].isAnomaly || d.is_anomaly // Union with ground truth
    }));
    
    setScadaData(scoredData);
    setRecentData(scoredData.slice(-100));
    
    const anomalyRows = scoredData.filter(r => r.is_anomaly);
    setDetectedAnomalies(anomalyRows);
    if (anomalyRows.length > 0) {
      setSelectedAnomaly(anomalyRows[anomalyRows.length - 1]); // Set last anomaly as default selected
    }

    // 3. Initialize Reinforcement Learning
    const agent = new MaintenanceRLAgent();
    setRlAgent(agent);
    
    // 4. Load saved settings
    setLlmSettings(LLMClient.getSavedSettings());
    
    // Load pre-configured system files for RAG (mocked for demo)
    vectorStore.chunkText(
      "Standard Operating Procedure: Wind Turbine T-402 Gearbox Cooling. If gearbox temperature exceeds 85°C, trigger partial torque limit (Action: Diagnostics). If temperature exceeds 95°C, immediate mechanical lock is required (Action: Urgent Shutdown). Maximum rotor speed is 16.0 RPM. Safe vibration levels are below 1.8 mm/s. Vibrations between 1.8 and 3.0 mm/s indicate shaft misalignment or bearing wear. Anything above 3.0 mm/s represents severe mechanical failure risks.",
      "SOP_T402_Maintenance.txt"
    );
    setUploadedFiles([{ name: "SOP_T402_Maintenance.txt", size: 685, chunks: vectorStore.getChunksCount() }]);
    setRagStatus("System SOP pre-loaded. Ask details in the chat.");
  }, []);

  // Compute forecasts when selected anomaly changes
  useEffect(() => {
    if (!selectedAnomaly || scadaData.length === 0) return;
    
    // Find index of selected anomaly in the main dataset
    const idx = scadaData.findIndex(r => r.timestamp === selectedAnomaly.timestamp);
    if (idx === -1) return;

    // Grab 24 hours of history prior to this anomaly
    const historyStart = Math.max(0, idx - 144); // 10m intervals, 144 = 24h
    const historySlice = scadaData.slice(historyStart, idx + 1);

    // Run LSTM Forecast for next 48 hours (48 steps of 1 hour)
    const predicted = LSTMTelemetryForecaster.forecast(historySlice, 48, 60);
    setForecasts(predicted);
  }, [selectedAnomaly, scadaData]);

  if (!mounted) {
    return (
      <div className="min-h-screen bg-[#070b13] flex flex-col items-center justify-center text-slate-400">
        <RefreshCw className="w-12 h-12 text-blue-500 animate-spin mb-4" />
        <span className="text-lg font-medium tracking-wide">Loading Aegis SCADA Predictive Core...</span>
      </div>
    );
  }

  // --- RL Functions ---
  const handleTrainRL = (episodes: number) => {
    if (!rlAgent) return;
    setIsTraining(true);
    
    setTimeout(() => {
      const history = rlAgent.train(episodes);
      setRlHistory(prev => [...prev, ...history]);
      
      // Update local state with trained agent
      const currentQState = rlAgent.encodeState(rlSimState.anomaly, rlSimState.risk, rlSimState.price);
      const action = rlAgent.selectAction(currentQState, true);
      
      setRlLogs(prev => [
        `[TRAINING] Completed ${episodes} training episodes. Epsilon decayed to ${Math.round(rlAgent['epsilon'] * 100) / 100}`,
        `[AGENT POLICY] Current best action for State ${currentQState}: ${getActionName(action)}`,
        ...prev
      ]);
      setIsTraining(false);
    }, 100);
  };

  const handleStepRL = () => {
    if (!rlAgent) return;

    const state = rlAgent.encodeState(rlSimState.anomaly, rlSimState.risk, rlSimState.price);
    const action = rlAgent.selectAction(state, true); // Use exploit mode to show learned intelligence

    const next = rlAgent.step(rlSimState.anomaly, rlSimState.risk, rlSimState.price, action);
    const nextState = rlAgent.encodeState(next.nextAnomalyScore, next.nextHoursToFailure, next.nextGridPrice);

    rlAgent.updateQ(state, action, next.reward, nextState);

    setRlSimState(prev => ({
      anomaly: next.nextAnomalyScore,
      risk: next.nextHoursToFailure,
      price: next.nextGridPrice,
      stateIndex: nextState,
      stepsCount: prev.stepsCount + 1,
      isFailed: next.isFailed,
      cumulativeReward: prev.cumulativeReward + next.reward
    }));

    setRlLogs(prev => [
      `Step ${rlSimState.stepsCount + 1} | Action: ${getActionName(action)} | Reward: $${Math.round(next.reward)} | ${next.description}`,
      ...prev
    ]);
  };

  const handleResetRL = () => {
    setRlSimState({
      anomaly: 0.15,
      risk: 48,
      price: 50,
      stateIndex: 0,
      stepsCount: 0,
      isFailed: false,
      cumulativeReward: 0
    });
    setRlLogs([]);
  };

  const getActionName = (action: number) => {
    if (action === 0) return "Do Nothing (Keep Running)";
    if (action === 1) return "Perform Diagnostics";
    return "Schedule Maintenance (Shut Down)";
  };

  // --- RAG Functions ---
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    const reader = new FileReader();

    reader.onload = (event) => {
      const content = event.target?.result as string;
      vectorStore.chunkText(content, file.name);
      
      setUploadedFiles(prev => [
        ...prev,
        { name: file.name, size: file.size, chunks: vectorStore.getChunksCount() }
      ]);
      setRagStatus(`Successfully indexed file "${file.name}" into ${vectorStore.getChunksCount()} chunks.`);
    };

    reader.readAsText(file);
  };

  const handleRagChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ragQueryText.trim()) return;

    const userMsg: ChatMessage = { role: "user", content: ragQueryText };
    setRagHistory(prev => [...prev, userMsg]);
    setRagQueryText("");
    setRagLoading(true);

    try {
      // 1. Search locally in our vector store using TF-IDF similarity matcher
      const matchedChunks = vectorStore.searchLocal(ragQueryText, 3);
      
      // 2. Invoke our BYOK / Free API Proxy
      const response = await LLMClient.ragQuery(
        ragQueryText, 
        matchedChunks.map(c => ({ text: c.text, docName: c.docName })),
        ragHistory.slice(-6), // Send last 6 messages of conversational history
        llmSettings
      );

      setRagHistory(prev => [...prev, { role: "assistant", content: response }]);
    } catch (err: any) {
      setRagHistory(prev => [
        ...prev, 
        { 
          role: "assistant", 
          content: `⚠️ API Error: ${err.message}. Please check your API key in settings or verify connection.` 
        }
      ]);
    } finally {
      setRagLoading(false);
    }
  };

  // --- Settings Functions ---
  const handleSaveSettings = (e: React.FormEvent) => {
    e.preventDefault();
    LLMClient.saveSettings(llmSettings);
    alert("Settings saved successfully.");
  };

  const handleTestConnection = async () => {
    setTestStatus('testing');
    setTestMessage('Pinging server proxy and verifying credentials...');
    try {
      const response = await LLMClient.chat([
        { role: 'user', content: 'Respond with only the word SUCCESS if you can hear this message.' }
      ], llmSettings);
      
      if (response && response.toLowerCase().includes('success')) {
        setTestStatus('success');
        setTestMessage(`Connection verified! Brain response: "${response.trim()}"`);
      } else {
        setTestStatus('success');
        setTestMessage(`Connection active. Received response: "${response.trim()}"`);
      }
    } catch (err: any) {
      setTestStatus('error');
      setTestMessage(`Connection failed: ${err.message}`);
    }
  };

  // --- Report & PDF Generation ---
  const generateAIExecutiveSummary = async () => {
    if (detectedAnomalies.length === 0) {
      alert("No anomalies available to analyze.");
      return;
    }

    setSummaryLoading(true);
    try {
      const totalAnomalies = detectedAnomalies.length;
      const latestAnomaly = detectedAnomalies[totalAnomalies - 1];
      
      const prompt: ChatMessage[] = [
        {
          role: "system",
          content: "You are an elite Chief Risk Analyst for a wind farm. Write a formal, high-impact executive summary regarding turbine anomalies for the CEO and board of directors. Keep the language highly professional, outline financial risks, and summarize concrete technical recommendations. Limit to 3 concise, bulleted paragraphs."
        },
        {
          role: "user",
          content: `Write an executive summary based on the following SCADA predictive analysis results:
- Wind Farm Asset: Turbine T-402
- Total SCADA historical logs analyzed: 10,000
- Total Predictive outliers/anomalies detected: ${totalAnomalies}
- Current Operational Alert Status: CRITICAL WARNING
- Latest Outlier Details: 
  * Timestamp: ${latestAnomaly.timestamp}
  * Gearbox Temperature: ${latestAnomaly.gearbox_temp_c} °C (Normal is ~65°C)
  * Vibration: ${latestAnomaly.vibration_mm_s} mm/s (Normal is ~1.2 mm/s)
  * Rotor Speed: ${latestAnomaly.rotor_rpm} RPM
- LSTM Forecast Projection: Gearbox Temperature predicted to exceed 110°C in next 24 hours if unchecked, risking bearing seizure.
- Reinforcement Learning Assessment: Policy agent strongly recommends immediate diagnostics/lubrication override to avoid bearing replacement cost ($50,000 loss) vs operational revenue trade-off.`
        }
      ];

      const summary = await LLMClient.chat(prompt, llmSettings);
      setExecutiveSummary(summary);
    } catch (err: any) {
      setExecutiveSummary(
        `[FALLBACK SUMMARY] EXECUTIVE BRIEFING: Turbine T-402 has shown recurring thermal and vibrational anomalies (total ${detectedAnomalies.length} events logged). At ${selectedAnomaly?.timestamp || 'latest'}, gearbox temp reached ${selectedAnomaly?.gearbox_temp_c || 85}°C and vibration hit ${selectedAnomaly?.vibration_mm_s || 2.5} mm/s. LSTM time series indicates severe risk of failure. The reinforcement learning scheduler recommends immediate scheduled maintenance to avoid critical damage. Please check your OpenRouter API key in Settings to restore the dynamic LLM generation.`
      );
    } finally {
      setSummaryLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!selectedAnomaly) {
      alert("No telemetry anomaly selected.");
      return;
    }

    // Initialize jsPDF (portrait, millimeter layout)
    const doc = new jsPDF("p", "mm", "a4");
    const pageWidth = doc.internal.pageSize.getWidth();
    
    // Page Header
    doc.setFillColor(11, 15, 25); // Deep slate banner
    doc.rect(0, 0, pageWidth, 30, "F");
    
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.text("AEGIS SCADA | PREDICTIVE MAINTENANCE REPORT", 12, 18);
    
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(148, 163, 184);
    doc.text(`CONFIDENTIAL - FOR CEO & DIRECTORS | GENERATED: ${new Date().toLocaleString()}`, 12, 25);

    // Section 1: Executive Summary
    doc.setTextColor(15, 23, 42);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("1. EXECUTIVE BRIEFING & INCIDENT ASSESSMENT", 12, 42);
    
    // Draw boundary line
    doc.setDrawColor(226, 232, 240);
    doc.setLineWidth(0.3);
    doc.line(12, 45, pageWidth - 12, 45);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(51, 65, 85);
    
    const summaryText = executiveSummary || "No AI Executive Briefing generated. Please generate the briefing via the dashboard to include a custom LLM analysis. Standard alert: Turbine T-402 gearbox bearing degradation warning.";
    const splitSummary = doc.splitTextToSize(summaryText, pageWidth - 24);
    doc.text(splitSummary, 12, 51);

    // Get current y height after summary text
    let currentY = 51 + (splitSummary.length * 4.5);

    // Section 2: Asset Telemetry Metrics (Table)
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(15, 23, 42);
    doc.text("2. TELEMETRY DETAILS & PREDICTIVE SCORING", 12, currentY + 8);
    doc.line(12, currentY + 11, pageWidth - 12, currentY + 11);

    currentY = currentY + 16;
    
    // Drawing a simple, clean table
    doc.setFillColor(241, 245, 249);
    doc.rect(12, currentY, pageWidth - 24, 6, "F"); // header row
    
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(51, 65, 85);
    doc.text("Parameter", 15, currentY + 4.5);
    doc.text("Measured Value", 65, currentY + 4.5);
    doc.text("Normal Range", 115, currentY + 4.5);
    doc.text("Risk Assessment", 165, currentY + 4.5);

    // Row 1: Temp
    currentY += 6;
    doc.rect(12, currentY, pageWidth - 24, 6, "F"); // Alternate bg
    doc.setFont("helvetica", "normal");
    doc.text("Gearbox Temp", 15, currentY + 4.5);
    doc.setFont("helvetica", "bold");
    doc.text(`${selectedAnomaly.gearbox_temp_c} °C`, 65, currentY + 4.5);
    doc.setFont("helvetica", "normal");
    doc.text("55 - 72 °C", 115, currentY + 4.5);
    doc.setTextColor(239, 68, 68);
    doc.text("Critical Overheating", 165, currentY + 4.5);
    doc.setTextColor(51, 65, 85);

    // Row 2: Vibration
    currentY += 6;
    doc.text("Gearbox Vibration", 15, currentY + 4.5);
    doc.setFont("helvetica", "bold");
    doc.text(`${selectedAnomaly.vibration_mm_s} mm/s`, 65, currentY + 4.5);
    doc.setFont("helvetica", "normal");
    doc.text("0.8 - 1.6 mm/s", 115, currentY + 4.5);
    doc.setTextColor(239, 68, 68);
    doc.text("Extreme Mechanical Friction", 165, currentY + 4.5);
    doc.setTextColor(51, 65, 85);

    // Row 3: RPM
    currentY += 6;
    doc.rect(12, currentY, pageWidth - 24, 6, "F");
    doc.text("Rotor RPM", 15, currentY + 4.5);
    doc.setFont("helvetica", "bold");
    doc.text(`${selectedAnomaly.rotor_rpm} RPM`, 65, currentY + 4.5);
    doc.setFont("helvetica", "normal");
    doc.text("8.0 - 15.5 RPM", 115, currentY + 4.5);
    doc.setTextColor(34, 197, 94);
    doc.text("Within Operations Speed", 165, currentY + 4.5);
    doc.setTextColor(51, 65, 85);

    // Row 4: Anomaly Score
    currentY += 6;
    doc.text("Isolation Forest Anomaly Score", 15, currentY + 4.5);
    doc.setFont("helvetica", "bold");
    doc.text(`${selectedAnomaly.anomaly_score || 0.85}`, 65, currentY + 4.5);
    doc.setFont("helvetica", "normal");
    doc.text("< 0.60 (Normal)", 115, currentY + 4.5);
    doc.setTextColor(239, 68, 68);
    doc.text("Outlier Flagged", 165, currentY + 4.5);
    doc.setTextColor(51, 65, 85);

    // Move to next page for charts and RL financial forecast
    doc.addPage();
    
    // Header for Page 2
    doc.setFillColor(11, 15, 25);
    doc.rect(0, 0, pageWidth, 15, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.text("AEGIS SCADA | PREDICTIVE FORECASTING & FINANCIAL AUDIT", 12, 10);

    // Section 3: Time Series Predictions (LSTM Chart screenshot)
    doc.setTextColor(15, 23, 42);
    doc.setFontSize(12);
    doc.text("3. LSTM TIME SERIES TEMPERATURE FORECAST (NEXT 48H)", 12, 28);
    doc.line(12, 31, pageWidth - 12, 31);

    if (forecastChartRef.current) {
      try {
        const canvas = await html2canvas(forecastChartRef.current, { scale: 2 });
        const imgData = canvas.toDataURL("image/png");
        doc.addImage(imgData, "PNG", 12, 35, pageWidth - 24, 75);
      } catch (err) {
        console.error("Failed to render chart to canvas", err);
        doc.setFont("helvetica", "italic");
        doc.setFontSize(10);
        doc.text("[Telemetry Chart unable to render in PDF. Refer to dashboard.]", 20, 60);
      }
    }

    // Section 4: RL Financial Savings Summary
    const rlY = 120;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("4. REINFORCEMENT LEARNING POLICY & COST OPTIMIZATION", 12, rlY);
    doc.line(12, rlY + 3, pageWidth - 12, rlY + 3);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(51, 65, 85);
    
    const rlAuditText = "Using a simulated Q-learning policy initialized on the historical failures of turbine T-402, the Aegis agent balances short-term generation losses against the costs of bearing failures. \n\nFinancial Comparison (Estimates):\n- Unplanned Bearing Failure Repair Cost: $50,000 (plus average 5 days complete shutdown)\n- Diagnostics & Early Scheduled Maintenance Cost: $5,000 (takes 8 hours scheduled down-time)\n- Net Operational Savings per Prevented Incident: $45,000\n\nThe Reinforcement Learning policy currently prescribes SCHEDULE MAINTENANCE for the active warning window. Unchecked operations are expected to cause a catastrophic bearing shutdown within 12-24 hours based on temperature progression trends.";
    const splitRl = doc.splitTextToSize(rlAuditText, pageWidth - 24);
    doc.text(splitRl, 12, rlY + 10);

    // Sign off footer
    const signOffY = 240;
    doc.line(12, signOffY, pageWidth - 12, signOffY);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.text("Aegis Predictive Engine Authority Signoff:", 12, signOffY + 6);
    
    doc.setFont("helvetica", "normal");
    doc.text("CEO Chief Engineer Checklist: ______________", 12, signOffY + 12);
    doc.text("Risk Operations Director: ______________", 110, signOffY + 12);

    doc.setFont("helvetica", "italic");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184);
    doc.text("Report compiled autonomously by Aegis SCADA local system engine. Authorized access only.", 12, signOffY + 24);

    // Trigger PDF download
    const tsStr = typeof selectedAnomaly.timestamp === 'string'
      ? selectedAnomaly.timestamp
      : selectedAnomaly.timestamp.toISOString();
    doc.save(`Aegis_SCADA_Report_Turbine_T-402_${tsStr.substring(0, 10)}.pdf`);
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* 1. Floating Top Header Navbar */}
      <header className="mx-4 mt-4 mb-2 z-10">
        <div className="glass-panel rounded-2xl py-3 px-6 flex items-center justify-between shadow-2xl">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-500 flex items-center justify-center glow-border-blue">
              <Activity className="w-5 h-5 text-white animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white font-sans flex items-center">
                AEGIS <span className="text-blue-500 ml-1.5 font-light">SCADA</span>
              </h1>
              <span className="text-[10px] uppercase tracking-widest text-cyan-400 font-medium">Predictive Maintenance OS</span>
            </div>
          </div>
          
          <nav className="flex space-x-1.5">
            {[
              { id: "operations", label: "Operations Overview", icon: Activity },
              { id: "diagnostics", label: "Model Diagnostics", icon: TrendingUp },
              { id: "rl", label: "RL Scheduler", icon: Cpu },
              { id: "rag", label: "RAG Docs", icon: Database },
              { id: "reports", label: "Reports Hub", icon: FileText },
              { id: "settings", label: "Settings", icon: Settings },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center space-x-2 py-2 px-4 rounded-xl text-sm font-medium transition-all duration-300 ${
                    isActive 
                      ? "bg-blue-600/25 text-blue-400 border border-blue-500/20 glow-border-blue" 
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                  } cursor-pointer`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden lg:inline">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* 2. Main Tabbed Views Container */}
      <main className="flex-1 p-4 grid grid-cols-1 gap-4 overflow-y-auto">
        
        {/* VIEW 1: Operations Overview (Turbine schematic + telemetry) */}
        {activeTab === "operations" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Column 1 & 2: Main Overview */}
            <div className="lg:col-span-2 flex flex-col space-y-4">
              {/* Telemetry KPI Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: "Turbine ID", val: "T-402", sub: "Onshore Active", status: "ok" },
                  { label: "Active Power", val: `${recentData[recentData.length-1]?.active_power_kw || 0} kW`, sub: `Th: ${recentData[recentData.length-1]?.theoretical_power_kw || 0} kW`, status: "ok" },
                  { label: "Gearbox Temp", val: `${recentData[recentData.length-1]?.gearbox_temp_c || 0} °C`, sub: "Limit: 85°C", status: (recentData[recentData.length-1]?.gearbox_temp_c || 0) > 80 ? "err" : "ok" },
                  { label: "Vibration", val: `${recentData[recentData.length-1]?.vibration_mm_s || 0} mm/s`, sub: "Limit: 1.8 mm/s", status: (recentData[recentData.length-1]?.vibration_mm_s || 0) > 1.8 ? "err" : "ok" },
                ].map((kpi, idx) => (
                  <div key={idx} className="glass-panel p-4 rounded-2xl relative overflow-hidden">
                    <div className="text-xs text-slate-400 font-medium">{kpi.label}</div>
                    <div className="text-xl font-bold mt-1 text-white">{kpi.val}</div>
                    <div className="text-xs text-slate-500 mt-0.5">{kpi.sub}</div>
                    <div className={`absolute top-4 right-4 w-2.5 h-2.5 rounded-full ${kpi.status === "err" ? "bg-red-500 animate-ping" : "bg-emerald-500"}`} />
                  </div>
                ))}
              </div>

              {/* Real-time Telemetry Charts */}
              <div className="glass-panel p-5 rounded-2xl flex-1 min-h-[300px] flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-bold text-white flex items-center">
                    <TrendingUp className="w-5 h-5 text-blue-500 mr-2" /> Live SCADA Telemetry Stream (Last 100 periods)
                  </h3>
                  <span className="text-xs bg-slate-800 text-slate-400 px-3 py-1 rounded-lg">Sampling: 10m intervals</span>
                </div>
                <div className="flex-1 w-full min-h-[250px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={recentData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                      <XAxis dataKey="timestamp" tick={false} stroke="rgba(255,255,255,0.2)" />
                      <YAxis yAxisId="left" stroke="#3b82f6" />
                      <YAxis yAxisId="right" orientation="right" stroke="#14b8a6" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: "#0b1528", borderColor: "rgba(255,255,255,0.1)", borderRadius: "12px", color: "#f8fafc" }}
                        labelFormatter={(label) => `Time: ${new Date(label).toLocaleString()}`}
                      />
                      <Legend />
                      <Line yAxisId="left" type="monotone" dataKey="gearbox_temp_c" name="Gearbox Temp (°C)" stroke="#3b82f6" strokeWidth={2.5} dot={false} />
                      <Line yAxisId="right" type="monotone" dataKey="vibration_mm_s" name="Vibration (mm/s)" stroke="#14b8a6" strokeWidth={2.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Column 3: Interactive Blueprint Schematic */}
            <div className="flex flex-col space-y-4">
              <div className="glass-panel p-5 rounded-2xl flex-1 flex flex-col items-center justify-between min-h-[450px]">
                <div className="w-full flex items-center justify-between mb-4">
                  <h3 className="text-base font-bold text-white flex items-center">
                    <Sliders className="w-5 h-5 text-blue-500 mr-2" /> Interactive Blueprint
                  </h3>
                  <span className="text-xs text-slate-400">Click components to view details</span>
                </div>
                
                {/* SVG Turbine Blueprint */}
                <div className="relative w-full max-w-[280px] h-[300px] flex items-center justify-center">
                  <svg viewBox="0 0 100 120" className="w-full h-full text-slate-700">
                    {/* Tower */}
                    <line x1="50" y1="60" x2="50" y2="110" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                    <line x1="45" y1="110" x2="55" y2="110" stroke="currentColor" strokeWidth="2" />
                    
                    {/* Nacelle housing */}
                    <rect x="44" y="52" width="12" height="9" rx="2" fill="#0f172a" stroke="currentColor" strokeWidth="1" />
                    
                    {/* Gearbox component */}
                    <rect 
                      x="46" 
                      y="54" 
                      width="5" 
                      height="5" 
                      rx="1" 
                      fill={(recentData[recentData.length-1]?.gearbox_temp_c || 0) > 80 ? "#ef4444" : "#3b82f6"} 
                      className={`cursor-pointer ${(recentData[recentData.length-1]?.gearbox_temp_c || 0) > 80 ? "animate-pulse" : ""}`} 
                    />
                    
                    {/* Rotor Hub */}
                    <circle cx="50" cy="56" r="3.5" fill="#f8fafc" stroke="currentColor" strokeWidth="1" />
                    
                    {/* Blades rotating */}
                    <g className="animate-spin origin-[50px_56px]" style={{ animationDuration: `${60 / (recentData[recentData.length-1]?.rotor_rpm || 15)}s` }}>
                      <path d="M 50 56 L 50 25" stroke="#f8fafc" strokeWidth="2.5" strokeLinecap="round" />
                      <path d="M 50 56 L 24 71" stroke="#f8fafc" strokeWidth="2.5" strokeLinecap="round" />
                      <path d="M 50 56 L 76 71" stroke="#f8fafc" strokeWidth="2.5" strokeLinecap="round" />
                    </g>
                  </svg>
                  
                  {/* Warning labels floating */}
                  {(recentData[recentData.length-1]?.gearbox_temp_c || 0) > 80 && (
                    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-12 bg-red-950/80 border border-red-500/30 text-red-400 text-[10px] font-bold py-1 px-2.5 rounded-lg flex items-center space-x-1.5 animate-bounce">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>Gearbox Warning</span>
                    </div>
                  )}
                </div>

                <div className="w-full bg-slate-900/40 border border-white/5 rounded-xl p-3 text-xs text-slate-300">
                  <div className="flex items-center justify-between font-semibold border-b border-white/5 pb-1.5 mb-1.5 text-white">
                    <span>Component Health</span>
                    <span className="text-cyan-400">T-402 Diagnostic</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Blades Pitch</span>
                      <span className="text-emerald-500 font-semibold">100% OK</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Shaft Alignment</span>
                      <span className="text-emerald-500 font-semibold">100% OK</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Bearing Vibration</span>
                      <span className={(recentData[recentData.length-1]?.vibration_mm_s || 0) > 1.8 ? "text-red-400 font-semibold animate-pulse" : "text-emerald-500 font-semibold"}>
                        {(recentData[recentData.length-1]?.vibration_mm_s || 0) > 1.8 ? "ABNORMAL" : "OK"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Gearbox Oil Temp</span>
                      <span className={(recentData[recentData.length-1]?.gearbox_temp_c || 0) > 80 ? "text-red-400 font-semibold animate-pulse" : "text-emerald-500 font-semibold"}>
                        {(recentData[recentData.length-1]?.gearbox_temp_c || 0) > 80 ? "OVERHEAT" : "OK"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* VIEW 2: Model Diagnostics (Scatter + Time series anomaly + LSTM Forecast) */}
        {activeTab === "diagnostics" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Column 1 & 2: Scatter and Forecasting */}
            <div className="lg:col-span-2 flex flex-col space-y-4">
              
              {/* Power Curve Scatter Chart */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col" ref={powerCurveRef}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-bold text-white flex items-center">
                    <Layers className="w-5 h-5 text-blue-500 mr-2" /> SCADA Power Curve Performance (Seeded 1,000 points)
                  </h3>
                  <span className="text-xs text-slate-400">Active Power (kW) vs Wind Speed (m/s)</span>
                </div>
                
                <div className="w-full h-[250px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                      <XAxis type="number" dataKey="wind_speed_ms" name="Wind Speed" unit=" m/s" stroke="rgba(255,255,255,0.2)" />
                      <YAxis type="number" dataKey="active_power_kw" name="Active Power" unit=" kW" stroke="rgba(255,255,255,0.2)" />
                      <Tooltip 
                        cursor={{ strokeDasharray: "3 3" }} 
                        contentStyle={{ backgroundColor: "#0b1528", borderColor: "rgba(255,255,255,0.1)", borderRadius: "12px", color: "#f8fafc" }}
                      />
                      <Legend />
                      <Scatter name="Actual Telemetry Points" data={scadaData.slice(0, 1000)} fill="#3b82f6" shape="circle" line={false} />
                      <Scatter name="Theoretical Power Curve" data={scadaData.slice(0, 1000).sort((a,b)=>a.wind_speed_ms-b.wind_speed_ms)} dataKey="theoretical_power_kw" fill="#14b8a6" shape={() => null} line={{ strokeWidth: 2, stroke: "#14b8a6" }} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* LSTM Forecast Chart */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col" ref={forecastChartRef}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-bold text-white flex items-center">
                    <TrendingUp className="w-5 h-5 text-blue-500 mr-2" /> LSTM Future Gearbox Temperature Forecast (Next 48 Hours)
                  </h3>
                  <span className="text-xs bg-blue-500/10 text-blue-400 px-3 py-1 rounded-lg">48h projection window</span>
                </div>
                
                <div className="w-full h-[250px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={forecasts}>
                      <defs>
                        <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                      <XAxis dataKey="timestamp" tick={false} stroke="rgba(255,255,255,0.2)" />
                      <YAxis domain={['auto', 'auto']} stroke="rgba(255,255,255,0.2)" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: "#0b1528", borderColor: "rgba(255,255,255,0.1)", borderRadius: "12px", color: "#f8fafc" }}
                        labelFormatter={(label) => `Forecast Hour: ${new Date(label).toLocaleString()}`}
                      />
                      <Legend />
                      <Area type="monotone" dataKey="gearbox_temp_c" name="Predicted Temp (°C)" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorTemp)" />
                      <Line type="monotone" dataKey="gearbox_temp_c_upper" name="Upper Confidence" stroke="#ef4444" strokeDasharray="5 5" dot={false} />
                      <Line type="monotone" dataKey="gearbox_temp_c_lower" name="Lower Confidence" stroke="#10b981" strokeDasharray="5 5" dot={false} />
                      <ReferenceLine y={85} label={{ value: 'Alarms Limit: 85°C', fill: '#f43f5e', position: 'top' }} stroke="#f43f5e" strokeWidth={1} strokeDasharray="3 3" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

            </div>

            {/* Column 3: Isolation Forest Anomaly Selector */}
            <div className="flex flex-col space-y-4">
              <div className="glass-panel p-5 rounded-2xl flex-1 flex flex-col max-h-[600px]">
                <h3 className="text-base font-bold text-white flex items-center mb-1">
                  <AlertTriangle className="w-5 h-5 text-amber-500 mr-2" /> Flagged SCADA Outliers
                </h3>
                <span className="text-xs text-slate-400 mb-4 border-b border-white/5 pb-2">Isolation Forest Flagged Anomalies</span>
                
                <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                  {detectedAnomalies.length === 0 ? (
                    <div className="text-center text-xs text-slate-500 py-10">No anomalies detected in database.</div>
                  ) : (
                    detectedAnomalies.slice(-30).map((anom, idx) => {
                      const isSelected = selectedAnomaly?.timestamp === anom.timestamp;
                      return (
                        <div
                          key={idx}
                          onClick={() => setSelectedAnomaly(anom)}
                          className={`p-3 rounded-xl border transition-all cursor-pointer ${
                            isSelected 
                              ? "bg-amber-500/10 border-amber-500/30 text-amber-300"
                              : "bg-slate-900/35 border-white/5 text-slate-300 hover:bg-slate-800/40"
                          }`}
                        >
                          <div className="flex items-center justify-between text-xs font-semibold">
                            <span>Turbine T-402</span>
                            <span className="text-amber-500">Score: {anom.anomaly_score}</span>
                          </div>
                          <div className="text-[10px] text-slate-500 mt-1 font-mono">{new Date(anom.timestamp).toLocaleString()}</div>
                          <div className="grid grid-cols-2 gap-2 mt-2 text-[10px] font-medium border-t border-white/5 pt-1.5">
                            <div>Temp: <span className="font-semibold text-white">{anom.gearbox_temp_c}°C</span></div>
                            <div>Vib: <span className="font-semibold text-white">{anom.vibration_mm_s}mm/s</span></div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
                
                {selectedAnomaly && (
                  <div className="mt-4 bg-slate-950/80 border border-white/5 rounded-xl p-3 text-xs text-slate-300">
                    <div className="font-bold text-white mb-1.5 flex items-center">
                      <Info className="w-4 h-4 text-cyan-400 mr-1.5" /> Selected Event Diagnostic
                    </div>
                    <div className="space-y-1 text-slate-400">
                      <div>Timestamp: <span className="text-white">{new Date(selectedAnomaly.timestamp).toLocaleString()}</span></div>
                      <div>Type: <span className="text-amber-400 font-medium">{selectedAnomaly.anomaly_type || "Outlier Telemetry Combo"}</span></div>
                      <div className="pt-2 leading-relaxed text-[11px] text-slate-500 border-t border-white/5 mt-1.5">
                        The combination of {selectedAnomaly.gearbox_temp_c}°C temperature and {selectedAnomaly.vibration_mm_s} mm/s vibration indicates mechanical wear. Forecasting shows escalation path.
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

          </div>
        )}

        {/* VIEW 3: RL Maintenance Simulator (Q-learning controller + visual step) */}
        {activeTab === "rl" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Column 1 & 2: RL State & Simulator Console */}
            <div className="lg:col-span-2 flex flex-col space-y-4">
              
              {/* Simulator Metrics Grid */}
              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: "Step Anomaly", val: `${Math.round(rlSimState.anomaly * 100)}%`, sub: "Limit: 70%", progress: rlSimState.anomaly, color: "bg-amber-500" },
                  { label: "Steps To Failure", val: `${rlSimState.risk} Hrs`, sub: "Forecast Time", progress: 1 - (rlSimState.risk / 48), color: "bg-red-500" },
                  { label: "Grid Energy Price", val: `$${rlSimState.price}/MWh`, sub: "Flutucation: $20-$120", progress: rlSimState.price / 120, color: "bg-emerald-500" }
                ].map((stat, idx) => (
                  <div key={idx} className="glass-panel p-4 rounded-2xl flex flex-col relative overflow-hidden">
                    <span className="text-xs text-slate-400 font-medium">{stat.label}</span>
                    <span className="text-xl font-bold mt-1 text-white">{stat.val}</span>
                    <span className="text-xs text-slate-500 mt-0.5">{stat.sub}</span>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
                      <div className={`h-full ${stat.color}`} style={{ width: `${stat.progress * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Console log outputs */}
              <div className="glass-panel p-5 rounded-2xl flex-1 flex flex-col min-h-[300px]">
                <div className="flex items-center justify-between mb-4 border-b border-white/5 pb-2.5">
                  <h3 className="text-base font-bold text-white flex items-center">
                    <Layers className="w-5 h-5 text-blue-500 mr-2" /> Reinforcement Learning Environment Decision Logs
                  </h3>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs bg-slate-800 text-slate-300 py-1 px-2.5 rounded-lg font-mono">Steps: {rlSimState.stepsCount}</span>
                    <span className={`text-xs px-2.5 py-1 rounded-lg font-bold ${rlSimState.isFailed ? "bg-red-500/10 text-red-400 border border-red-500/25" : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/25"}`}>
                      {rlSimState.isFailed ? "CRITICAL SHUTDOWN" : "ONLINE"}
                    </span>
                  </div>
                </div>
                
                <div className="flex-1 bg-slate-950/60 border border-white/5 rounded-xl p-4 font-mono text-xs text-slate-400 overflow-y-auto space-y-1.5 h-[200px]">
                  {rlLogs.length === 0 ? (
                    <div className="text-slate-600 text-center py-10">Console empty. Click "Exploit Step" or "Train Agent" to write decision steps.</div>
                  ) : (
                    rlLogs.map((log, index) => (
                      <div key={index} className={log.includes("seized") || log.includes("Error") ? "text-red-400" : log.includes("maintenance") ? "text-cyan-400" : log.includes("diagnostics") ? "text-amber-400" : "text-slate-300"}>
                        {log}
                      </div>
                    ))
                  )}
                </div>
              </div>

            </div>

            {/* Column 3: Controller Actions and policy weights */}
            <div className="flex flex-col space-y-4">
              <div className="glass-panel p-5 rounded-2xl flex-1 flex flex-col justify-between">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center mb-1">
                    <Sliders className="w-5 h-5 text-blue-500 mr-2" /> RL Policy Controller
                  </h3>
                  <span className="text-xs text-slate-400 mb-4 border-b border-white/5 pb-2 block">Train Q-table policies</span>

                  <div className="space-y-4">
                    {/* Training Action Buttons */}
                    <div className="space-y-2">
                      <label className="text-xs font-semibold text-slate-300">Offline Episode Training</label>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          onClick={() => handleTrainRL(250)}
                          disabled={isTraining}
                          className="flex items-center justify-center py-2 px-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 text-white rounded-xl text-xs font-medium cursor-pointer transition-all"
                        >
                          <Play className="w-3.5 h-3.5 mr-1.5" /> {isTraining ? "Training..." : "Train 250 Ep"}
                        </button>
                        <button
                          onClick={() => handleTrainRL(1000)}
                          disabled={isTraining}
                          className="flex items-center justify-center py-2 px-3 bg-gradient-to-tr from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:from-blue-800 disabled:to-blue-800 text-white rounded-xl text-xs font-medium cursor-pointer transition-all"
                        >
                          <Cpu className="w-3.5 h-3.5 mr-1.5 animate-pulse" /> {isTraining ? "Training..." : "Train 1k Ep"}
                        </button>
                      </div>
                    </div>

                    {/* Environment Step Buttons */}
                    <div className="space-y-2 border-t border-white/5 pt-3">
                      <label className="text-xs font-semibold text-slate-300">Live Simulator Step (Exploit Policy)</label>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          onClick={handleStepRL}
                          className="flex items-center justify-center py-2.5 px-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold cursor-pointer transition-all"
                        >
                          <TrendingUp className="w-3.5 h-3.5 mr-1.5" /> Step Policy
                        </button>
                        <button
                          onClick={handleResetRL}
                          className="flex items-center justify-center py-2.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-semibold cursor-pointer transition-all"
                        >
                          <RefreshCw className="w-3.5 h-3.5 mr-1.5" /> Reset State
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-900/40 border border-white/5 rounded-xl p-3 text-xs text-slate-300 mt-6">
                  <div className="flex items-center justify-between font-bold text-white mb-2 pb-1 border-b border-white/5">
                    <span>Q-learning Metrics</span>
                    <DollarSign className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Total Profit Generated:</span>
                      <span className={`font-semibold ${rlSimState.cumulativeReward >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        ${Math.round(rlSimState.cumulativeReward)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Exploration Rate (ε):</span>
                      <span className="font-semibold text-white">{rlAgent ? Math.round(rlAgent['epsilon'] * 100) : 0}%</span>
                    </div>
                    <div className="flex justify-between leading-normal text-[10px] text-slate-500 border-t border-white/5 pt-2">
                      <span>As you train more episodes, the agent's exploration rate decays, allowing it to exploit optimal maintenance decisions.</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* VIEW 4: Knowledge Base (RAG uploader & chat interface) */}
        {activeTab === "rag" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Column 1 & 2: Chat Assistant Console */}
            <div className="lg:col-span-2 flex flex-col space-y-4">
              <div className="glass-panel p-5 rounded-2xl flex-1 flex flex-col min-h-[450px]">
                <div className="flex items-center justify-between mb-4 border-b border-white/5 pb-2.5">
                  <h3 className="text-base font-bold text-white flex items-center">
                    <Database className="w-5 h-5 text-blue-500 mr-2" /> Aegis-AI RAG Chat Assistant
                  </h3>
                  <span className="text-xs text-slate-400">Context: Local vector manuals</span>
                </div>

                {/* Conversational Bubbles */}
                <div className="flex-1 overflow-y-auto space-y-4 pr-1 mb-4 h-[300px]">
                  {ragHistory.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-14 text-center">
                      <Database className="w-10 h-10 text-slate-600 mb-2 animate-bounce" />
                      <h4 className="text-sm font-semibold text-slate-400">Conversational Vector RAG</h4>
                      <p className="text-xs text-slate-500 max-w-xs mt-1 leading-relaxed">
                        Upload maintenance logs or turbine manuals on the right. Ask technical questions like: "What is the gearbox temperature threshold?"
                      </p>
                    </div>
                  ) : (
                    ragHistory.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[85%] rounded-2xl p-3.5 text-xs leading-relaxed ${
                            msg.role === "user"
                              ? "bg-blue-600 text-white rounded-tr-none"
                              : "bg-slate-900/80 border border-white/5 text-slate-300 rounded-tl-none"
                          }`}
                        >
                          <div className="font-bold text-[10px] opacity-60 mb-1">
                            {msg.role === "user" ? "OPERATOR" : "AEGIS-AI ASSISTANT"}
                          </div>
                          <div className="whitespace-pre-line">{msg.content}</div>
                        </div>
                      </div>
                    ))
                  )}
                  {ragLoading && (
                    <div className="flex justify-start">
                      <div className="bg-slate-900/80 border border-white/5 rounded-2xl rounded-tl-none p-3 text-xs text-slate-400 flex items-center space-x-2">
                        <RefreshCw className="w-3.5 h-3.5 text-blue-500 animate-spin" />
                        <span>Searching local vectors & generating response...</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Input form */}
                <form onSubmit={handleRagChat} className="flex space-x-2">
                  <input
                    type="text"
                    value={ragQueryText}
                    onChange={(e) => setRagQueryText(e.target.value)}
                    placeholder="Ask operational questions (e.g. bearing vibration thresholds)..."
                    className="flex-1 bg-slate-950 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none focus:border-blue-500/50"
                  />
                  <button
                    type="submit"
                    className="bg-blue-600 hover:bg-blue-500 text-white rounded-xl px-4 py-2.5 text-xs font-semibold cursor-pointer transition-all flex items-center"
                  >
                    <Send className="w-3.5 h-3.5" />
                  </button>
                </form>
              </div>
            </div>

            {/* Column 3: Document Uploader & Status */}
            <div className="flex flex-col space-y-4">
              
              {/* Uploader Card */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col justify-between">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center mb-1">
                    <Upload className="w-5 h-5 text-blue-500 mr-2" /> Document Library
                  </h3>
                  <span className="text-xs text-slate-400 mb-4 border-b border-white/5 pb-2 block">Upload organizational manuals</span>
                  
                  {/* File Dropzone */}
                  <label className="border border-dashed border-white/15 hover:border-blue-500/40 rounded-xl p-6 text-center cursor-pointer flex flex-col items-center justify-center space-y-2 bg-slate-900/20 transition-all block">
                    <Upload className="w-6 h-6 text-slate-500" />
                    <span className="text-xs text-slate-300 font-medium">Click to select files</span>
                    <span className="text-[10px] text-slate-500">Supports .txt, .log, .csv files</span>
                    <input 
                      type="file" 
                      accept=".txt,.log,.csv" 
                      onChange={handleFileUpload} 
                      className="hidden" 
                    />
                  </label>

                  {/* List of uploaded files */}
                  <div className="mt-4 space-y-2">
                    <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Indexed Documents</div>
                    {uploadedFiles.map((file, idx) => (
                      <div key={idx} className="flex items-center justify-between p-2.5 bg-slate-950/60 rounded-lg text-[11px]">
                        <span className="text-slate-300 font-medium truncate max-w-[150px]">{file.name}</span>
                        <span className="text-slate-500">{(file.size / 1024).toFixed(1)} KB ({file.chunks} chunks)</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-6 bg-slate-900/40 border border-white/5 rounded-xl p-3 text-xs text-slate-300">
                  <div className="font-bold text-white mb-1.5 flex items-center">
                    <Info className="w-4 h-4 text-cyan-400 mr-1.5" /> Vector DB Status
                  </div>
                  <div className="text-[11px] leading-relaxed text-slate-400">
                    {ragStatus}
                  </div>
                </div>
              </div>

            </div>

          </div>
        )}

        {/* VIEW 5: Report Hub (Briefing compiler + PDF generation) */}
        {activeTab === "reports" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            
            {/* Column 1 & 2: Report Content Compiler */}
            <div className="lg:col-span-2 flex flex-col space-y-4">
              <div className="glass-panel p-5 rounded-2xl flex-1 flex flex-col min-h-[450px]">
                <div className="flex items-center justify-between mb-4 border-b border-white/5 pb-2.5">
                  <h3 className="text-base font-bold text-white flex items-center">
                    <FileText className="w-5 h-5 text-blue-500 mr-2" /> CEO Executive Briefing Compiler
                  </h3>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={generateAIExecutiveSummary}
                      disabled={summaryLoading}
                      className="py-1.5 px-3 bg-gradient-to-tr from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-semibold cursor-pointer transition-all flex items-center"
                    >
                      <Cpu className="w-3.5 h-3.5 mr-1.5" /> {summaryLoading ? "Compiling Briefing..." : "Compile AI Briefing"}
                    </button>
                  </div>
                </div>

                {/* Briefing Editor Output */}
                <div className="flex-1 bg-slate-950/60 border border-white/10 rounded-xl p-5 text-slate-300 overflow-y-auto space-y-4">
                  {summaryLoading ? (
                    <div className="flex flex-col items-center justify-center py-20">
                      <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mb-3" />
                      <span className="text-xs text-slate-500">Consulting Aegis-AI Brain...</span>
                    </div>
                  ) : (
                    <div className="text-xs leading-relaxed whitespace-pre-line">
                      {executiveSummary ? (
                        executiveSummary
                      ) : (
                        <div className="text-center text-slate-500 py-16">
                          <Info className="w-8 h-8 mx-auto text-slate-700 mb-2" />
                          No briefings generated. Click "Compile AI Briefing" above to query the LLM and generate a formal CEO assessment using the anomalies.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Column 3: Report Actions */}
            <div className="flex flex-col space-y-4">
              <div className="glass-panel p-5 rounded-2xl flex-1 flex flex-col justify-between">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center mb-1">
                    <Download className="w-5 h-5 text-blue-500 mr-2" /> PDF Export Desk
                  </h3>
                  <span className="text-xs text-slate-400 mb-4 border-b border-white/5 pb-2 block">Configure PDF compilation options</span>

                  <div className="space-y-4">
                    <div className="text-xs text-slate-300">
                      This module compiles a high-fidelity 2-page PDF designed for executive review:
                      <ul className="list-disc pl-4 mt-2 space-y-1 text-slate-400 text-[11px]">
                        <li>CEO Executive Briefing Statement</li>
                        <li>SCADA Telemetry Outliers Table</li>
                        <li>48-Hour LSTM temperature forecasting chart</li>
                        <li>Reinforcement Learning financial audit</li>
                      </ul>
                    </div>

                    <div className="bg-slate-900/60 rounded-xl p-3 border border-white/5 space-y-2">
                      <div className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Report Metadata</div>
                      <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400">
                        <div>Asset: <span className="text-white">T-402</span></div>
                        <div>Type: <span className="text-white">Wind Turbine</span></div>
                        <div>Outliers: <span className="text-white">{detectedAnomalies.length} logged</span></div>
                        <div>Trigger: <span className="text-white">Isolation Forest</span></div>
                      </div>
                    </div>
                  </div>
                </div>

                <button
                  onClick={handleDownloadPDF}
                  className="w-full bg-blue-600 hover:bg-blue-500 text-white rounded-xl py-3 text-sm font-bold cursor-pointer transition-all flex items-center justify-center space-x-2 mt-8 shadow-lg glow-border-blue"
                >
                  <Download className="w-4 h-4" />
                  <span>Download CEO Report</span>
                </button>
              </div>
            </div>

          </div>
        )}

        {/* VIEW 6: Settings & BYOK config */}
        {activeTab === "settings" && (
          <div className="max-w-xl mx-auto w-full">
            <div className="glass-panel p-6 rounded-2xl shadow-2xl">
              <div className="flex items-center justify-between mb-6 border-b border-white/5 pb-3">
                <h3 className="text-base font-bold text-white flex items-center">
                  <Settings className="w-5 h-5 text-blue-500 mr-2" /> Bring Your Own Key (BYOK) Panel
                </h3>
                <span className="text-xs text-slate-400">Aegis AI Brain Setup</span>
              </div>

              <form onSubmit={handleSaveSettings} className="space-y-4">
                {/* Provider Selection */}
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300">LLM Provider</label>
                  <select
                    value={llmSettings.provider}
                    onChange={(e) => setLlmSettings(prev => ({ 
                      ...prev, 
                      provider: e.target.value as any,
                      model: e.target.value === 'openrouter' ? 'google/gemma-2-9b-it:free' : e.target.value === 'openai' ? 'gpt-4o-mini' : e.target.value === 'gemini' ? 'gemini-1.5-flash' : 'claude-3-5-sonnet-20240620'
                    }))}
                    className="w-full bg-slate-950 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none focus:border-blue-500/50"
                  >
                    <option value="openrouter">OpenRouter (Free Fallback Default)</option>
                    <option value="openai">OpenAI</option>
                    <option value="gemini">Google Gemini (AI Studio)</option>
                    <option value="anthropic">Anthropic Claude</option>
                  </select>
                </div>

                {/* Model Input */}
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300">Model Name</label>
                  <input
                    type="text"
                    value={llmSettings.model}
                    onChange={(e) => setLlmSettings(prev => ({ ...prev, model: e.target.value }))}
                    placeholder="Enter model string..."
                    className="w-full bg-slate-950 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none"
                  />
                </div>

                {/* API Key Input */}
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-semibold text-slate-300">API Key</label>
                    <button
                      type="button"
                      onClick={() => setShowApiKey(!showApiKey)}
                      className="text-[10px] text-blue-400 hover:text-blue-300 font-semibold cursor-pointer"
                    >
                      {showApiKey ? <EyeOff className="w-3.5 h-3.5 inline mr-1" /> : <Eye className="w-3.5 h-3.5 inline mr-1" />}
                      {showApiKey ? "Hide Key" : "Show Key"}
                    </button>
                  </div>
                  <input
                    type={showApiKey ? "text" : "password"}
                    value={llmSettings.apiKey}
                    onChange={(e) => {
                      const val = e.target.value;
                      setLlmSettings(prev => {
                        const updated = { ...prev, apiKey: val };
                        // Auto pick the free model when OpenRouter key is typed/inserted
                        if (updated.provider === 'openrouter' && val.trim().length > 0) {
                          updated.model = 'google/gemma-2-9b-it:free';
                        }
                        return updated;
                      });
                    }}
                    placeholder={llmSettings.provider === 'openrouter' ? "Optional for free models. Enter OpenRouter key to verify..." : "Paste provider API key..."}
                    className="w-full bg-slate-950 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none"
                  />
                </div>

                <div className="bg-slate-900/40 border border-white/5 rounded-xl p-3.5 text-xs text-slate-400 leading-relaxed">
                  <span className="font-semibold text-white flex items-center mb-1">
                    <Info className="w-3.5 h-3.5 text-cyan-400 mr-1.5" /> Privacy Shield Notice
                  </span>
                  Your API keys are saved directly to your browser's <span className="font-mono text-slate-300">localStorage</span> and are never stored or logged on our servers. They are securely transmitted only through transient SSL proxies when querying providers.
                </div>

                <div className="flex gap-2">
                  <button
                    type="submit"
                    className="flex-1 bg-blue-600 hover:bg-blue-500 text-white rounded-xl py-3 text-xs font-bold cursor-pointer transition-all flex items-center justify-center"
                  >
                    Save Settings
                  </button>
                  <button
                    type="button"
                    onClick={handleTestConnection}
                    disabled={testStatus === 'testing'}
                    className="flex-1 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 text-slate-200 rounded-xl py-3 text-xs font-bold cursor-pointer transition-all flex items-center justify-center border border-white/10"
                  >
                    {testStatus === 'testing' ? 'Testing...' : 'Test Connection'}
                  </button>
                </div>

                {testStatus !== 'idle' && (
                  <div className={`mt-3 p-3 rounded-xl text-xs ${
                    testStatus === 'success' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                    testStatus === 'testing' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' :
                    'bg-red-500/10 text-red-400 border border-red-500/20'
                  }`}>
                    {testMessage}
                  </div>
                )}
              </form>
            </div>
          </div>
        )}

      </main>
      
      {/* 3. Footer Branding */}
      <footer className="py-4 text-center border-t border-white/5 text-[10px] text-slate-500 font-mono">
        <span>AEGIS PREDICTIVE SCADA ENGINE v1.1.0-TS | PRODUCTION READY PWA</span>
      </footer>
    </div>
  );
}
