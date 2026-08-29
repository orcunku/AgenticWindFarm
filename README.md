# ⚡ Wind Turbine SCADA Agentic AI & MCP Platform

An enterprise-ready industrial AI platform engineered for real-time wind farm telemetry monitoring, ML fault detection, Agentic RAG diagnostics via Model Context Protocol (MCP), and 2-Opt maintenance route optimization.

---

## 🚀 Live Demo & Showcase Links

* **Interactive Streamlit Web App:** https://agenticwindfarm-xcfp5xkrdekl2chdrjxraa.streamlit.app/
* **Bolt Interactive Portal:** https://wind-turbine-scada-a-4d92.bolt.host
* **GitHub Pages Landing Page:** https://orcunku.github.io/AgenticWindFarm/

---

## 🛠️ Key System Features

* **False Alarm Suppression Engine (>83% Noise Reduction):** Combines unsupervised Isolation Forest anomaly detection with supervised Random Forest classification to eliminate transient SCADA alarm fatigue.
* **Maintenance Truck Rollout Optimizer:** Compares Greedy Nearest-Neighbor heuristics against 2-Opt Travelling Salesperson Problem (TSP) local search to optimize field engineer routes.
* **Model Context Protocol (MCP) Integration:** Exposes standardized tool discovery endpoints (`/mcp/tools`) enabling LLM agents to execute real-time operational diagnostics and action plans.
* **Drone Surface Inspection Vision:** Simulates multi-modal computer vision blade analysis to detect micro-cracks and leading-edge erosion defects.
* **Real-Time Telemetry Streaming:** Dynamic WebSocket integration streaming 1-second generator sensor updates directly to interactive telemetry charts.

---
## 🏗️ System Architecture
┌───────────────────────────────────┐
                           │     Streamlit / Bolt UI           │
                           │   (Frontend Dashboard & Stream)   │
                           └─────────────────┬─────────────────┘
                                             │ REST / WebSockets
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FastAPI Backend                                            │
│                                                                                                  │
│   ┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐   │
│   │    Ensemble ML Filter    │   │   2-Opt TSP Optimizer    │   │  Model Context Protocol  │   │
│   │ (IsoForest + RandForest) │   │ (Field Service Routing)  │   │     (/mcp/tools API)     │   │
│   └──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

## 💻 Local Setup & Development

### 1. Clone the Repository
```bash
git clone [https://github.com/orcunku/AgenticWindFarm.git](https://github.com/orcunku/AgenticWindFarm.git)
cd AgenticWindFarm

pip install -r requirements.txt

uvicorn backend:app --port 8000

streamlit run app.py


Developer: Orcun Kusdemir

GitHub: @orcunku

                                             
