<div align="center">

<img src="https://img.shields.io/badge/STATUS-LIVE-00ff88?style=for-the-badge&labelColor=0d0d0d" />
<img src="https://img.shields.io/badge/LLM%20CALLS-41%20%40%20100%25%20SUCCESS-FFD700?style=for-the-badge&labelColor=0d0d0d" />
<img src="https://img.shields.io/badge/RENDER-DEPLOYED-46E3B7?style=for-the-badge&logo=render&logoColor=white&labelColor=0d0d0d" />

<br /><br />

```
██╗███╗   ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗██╗  ██╗
██║████╗  ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝╚██╗██╔╝
██║██╔██╗ ██║███████╗██║██║  ███╗███████║   ██║    ╚███╔╝ 
██║██║╚██╗██║╚════██║██║██║   ██║██╔══██║   ██║    ██╔██╗ 
██║██║ ╚████║███████║██║╚██████╔╝██║  ██║   ██║   ██╔╝ ██╗
╚═╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
```

### **AI Data Intelligence — Automated EDA, LLM Analytics & Executive PDF Reports**
*Raw data in. LLM-written insights out. No manual interpretation needed.*

<br />

[![Live App](https://img.shields.io/badge/%20Live%20App-insightx--ai--data--intelligence.onrender.com-FFD700?style=for-the-badge)](https://insightx-ai-data-intelligence.onrender.com)

</div>

---

## What Is InsightX?

InsightX is a **production-deployed analytics automation platform** that transforms raw datasets into executive-ready intelligence reports — automatically. Upload a CSV, and InsightX runs full EDA, generates LLM-written analytical narratives via Groq, and exports a formatted PDF report — all within milliseconds.

No dashboards to build. No manual analysis. No interpretation overhead.

---

## Performance Benchmarks

| Operation | Result |
|-----------|--------|
| Full Pipeline (2,823 rows × 25 cols) | **285–596 ms** |
| PDF Report Generation | **~332 ms** |
| LLM API Calls Validated | **41 calls** |
| LLM API Success Rate | **100%** |
| Avg LLM Response Time | **3.2 s** |
| Avg LLM Output Length | **1,354 characters** |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│              CSV / Dataset Upload (Streamlit UI)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROFILING MODULE                             │
│  Missing Value Analysis · Duplicate Detection · Outlier ID     │
│  Data Quality Scoring · Distribution Stats · Completeness %    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM INTERPRETATION MODULE                     │
│  Statistical Results → Groq Prompt Templates                   │
│  → Plain-Language Analytical Narratives (Llama 3.1 8B)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PDF GENERATION MODULE                          │
│  ReportLab → Stats + LLM Narrative + Visualizations           │
│  → Single Executive-Style PDF Report                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM Backend** | Groq API — Llama 3.1 8B |
| **Prompt Engineering** | Structured statistical → narrative templates |
| **PDF Generation** | ReportLab |
| **EDA Engine** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib |
| **Frontend** | Streamlit |
| **Deployment** | Render |
| **Language** | Python |

---

## Core Features

- **Full EDA automation** — profiling, missing values, duplicates, outliers, quality scoring
- **LLM-written narratives** — Groq Llama 3.1 converts statistics into plain-English insights
- **Executive PDF reports** — auto-generated, combines stats + narrative + charts in one file
- **Structured prompt templates** — engineered for consistent, structured LLM outputs
- **Modular pipeline** — profiling, interpretation, and PDF generation are fully independent
- **Validated at scale** — 41 LLM calls, 100% API success rate, consistent structured outputs
- **Sub-600ms pipeline** — full analysis on 2,823×25 dataset completes in under a second

---

## What Gets Analyzed

```
Dataset Profiling
├── Shape & Column Types
├── Missing Value Analysis (count + %)
├── Duplicate Row Detection
├── Outlier Identification (IQR method)
├── Data Quality Score (0–100)
├── Distribution Statistics (mean, std, skew, kurtosis)
└── Completeness Percentage per Column

LLM Interpretation
├── Plain-language summary of dataset health
├── Key anomalies and patterns flagged
├── Data quality recommendations
└── Executive-ready narrative output

PDF Report
├── Cover page with dataset metadata
├── Statistical tables
├── Visualizations (distributions, correlations)
└── LLM-written analytical narrative
```

---

## Run Locally

```bash
git clone https://github.com/sujanya-hub/InsightX-AI-Data-Intelligence
cd InsightX-AI-Data-Intelligence
pip install -r requirements.txt
```

Add your API key to `.env`:
```env
GROQ_API_KEY=your_groq_api_key
```

```bash
streamlit run app.py
```

---

## Project Structure

```
InsightX/
├── assets/
├── core/
├── utils/
├── outputs/
├── modules/
├── .streamlit/
├── .vscode/
├── .gitignore
├── __init__.py
├── app.py
└── requirements.txt
```

---

## Live Deployment

| Service | URL |
|---------|-----|
| **Live App** | [insightx-ai-data-intelligence.onrender.com](https://insightx-ai-data-intelligence.onrender.com) |

> *Render free-tier has ~7s cold-start on first request. Pipeline runs fast after warmup.*

---

<div align="center">

**Built by [Sujanya Srinivas](https://linkedin.com/in/sujanya-s-538a7a2b1)**
[LinkedIn](https://linkedin.com/in/sujanya-s-538a7a2b1) · [GitHub](https://github.com/sujanya-hub) · [Email](mailto:sujanyasrinivasa@gmail.com)

</div>
