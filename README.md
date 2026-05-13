# InsightX
### Modular AI Data Intelligence Platform

<p align="center">
  Automated analytics, data quality evaluation, AI-assisted interpretation, and executive report generation.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Analytics-ScikitLearn-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20Llama3.1-black?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Reporting-ReportLab-blue?style=for-the-badge" />
</p><div align="center">

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
├── app.py                  # Streamlit frontend
├── modules/
│   ├── profiler.py         # EDA & statistical analysis
│   ├── llm_interpreter.py  # Groq prompt templates & API calls
│   └── pdf_generator.py    # ReportLab report assembly
├── requirements.txt
└── .env.example
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

<p align="center">
  <a href="https://insightx-ai-data-intelligence.onrender.com">
    <img src="https://img.shields.io/badge/Live%20Demo-Render-success?style=for-the-badge" />
  </a>

  <a href="https://github.com/sujanya-hub/InsightX-AI-Data-Intelligence">
    <img src="https://img.shields.io/badge/GitHub-Repository-purple?style=for-the-badge" />
  </a>
</p>

---

## Live Deployment

| Service | Link |
|---|---|
| Live Demo | https://insightx-ai-data-intelligence.onrender.com |
| GitHub Repository | https://github.com/sujanya-hub/InsightX-AI-Data-Intelligence |

---

## Overview

InsightX is a modular AI-powered analytics platform built to automate the end-to-end data analysis workflow — from dataset ingestion and preprocessing to statistical interpretation and report generation.

The system combines:
- automated data profiling
- preprocessing pipelines
- exploratory analysis
- AI-assisted interpretation
- executive-style reporting

into a single workflow designed for rapid analytical review and decision support.

The project focuses on combining deterministic analytics with LLM-assisted explanations while keeping the analytical pipeline modular and explainable.

---

### Executive Reporting Interface

Automated visual summaries and PDF-ready analytical reporting workflows.

![Reporting Interface](assets/reporting-dashboard.png)

---

## System Workflow

InsightX simulates a real-world analytics pipeline through multiple processing stages:

1. Dataset ingestion and schema validation  
2. Data quality analysis and profiling  
3. Automated preprocessing and cleaning  
4. Exploratory Data Analysis (EDA)  
5. Statistical insight generation  
6. LLM-assisted interpretation  
7. Executive report generation  

The architecture separates analytics, AI reasoning, reporting, and UI layers so components can evolve independently.

---

## Core Modules

### Data Engineering & Quality Assurance

- Automated statistical profiling
- Missing value analysis
- Duplicate detection
- Outlier identification
- Data quality scoring
- Distribution analysis

### Analytics & AI Copilot

- Exploratory Data Analysis (EDA)
- KPI and trend extraction
- Scenario-based analytical reasoning
- Risk and opportunity identification
- LLM-assisted natural language explanations

### Reporting & Visualization

- Automated chart generation
- Business-style analytical summaries
- PDF report generation using ReportLab
- Structured export-ready outputs

---

## Why This Architecture?

Most analytics dashboards stop at visualization and require manual interpretation of statistical outputs.

InsightX extends this pipeline by combining:
- deterministic statistical analysis
- automated preprocessing
- AI-assisted interpretation
- structured reporting

to reduce manual analytical overhead while keeping the workflow explainable and modular.

The system intentionally separates:
- preprocessing logic
- analytics pipelines
- AI interpretation
- reporting generation

so the platform remains maintainable and extensible as additional analytical modules are added.

---

## Engineering Decisions

| Decision | Reasoning |
|---|---|
| Modular pipeline separation | Simplifies debugging and future expansion |
| Deterministic preprocessing | Improves reproducibility of analytical outputs |
| Data quality scoring heuristics | Provides quick dataset health estimation |
| LLM-assisted explanations | Converts statistical outputs into readable insights |
| ReportLab integration | Enables exportable executive-style reporting |
| Streamlit state management | Supports multi-module interaction and caching |

---

## System Architecture

```text
InsightX
│
├── core/
│   ├── ingestion/        # Data loading and validation
│   ├── cleaning/         # Preprocessing pipelines
│   ├── profiling/        # Data quality and statistics
│   ├── analytics/        # EDA and business logic
│   ├── ai/               # LLM integration and prompt handling
│   └── reporting/        # PDF generation
│
├── modules/              # Streamlit UI modules
├── utils/                # Shared utilities
└── app.py                # Application controller
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Analytics | Scikit-learn |
| LLM Integration | Groq API (Llama / Mixtral) |
| Visualization | Matplotlib, Seaborn |
| Reporting | ReportLab |
| Deployment | Render |

---

## Project Structure

```text
InsightX/
│
├── assets/
│   ├── data-quality-dashboard.png
│   ├── ai-copilot.png
│   └── reporting-dashboard.png
│
├── core/
│   ├── ingestion/
│   ├── cleaning/
│   ├── profiling/
│   ├── analytics/
│   ├── ai/
│   └── reporting/
│
├── modules/
├── utils/
├── app.py
├── requirements.txt
└── README.md
```

---

## Performance Notes

| Metric | Observation |
|---|---|
| Processing Flow | End-to-end automated analytics pipeline |
| Report Generation | Automated PDF export support |
| AI Interpretation | Real-time LLM-assisted insight generation |
| Scalability | Modular architecture for future expansion |

Performance depends on:
- dataset size
- preprocessing complexity
- visualization rendering load
- external LLM response latency

---

## Current Limitations

- Large datasets may increase preprocessing and visualization time.
- Outlier detection heuristics may require tuning for domain-specific datasets.
- AI-generated interpretations should still be reviewed for business-critical decisions.
- PDF report generation can become slower for visualization-heavy reports.
- Persistent storage and user session management are not yet implemented.

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/sujanya-hub/InsightX-AI-Data-Intelligence.git

cd InsightX-AI-Data-Intelligence
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create:

```bash
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_api_key_here"
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Deployment (Render)

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Environment Variable

```env
GROQ_API_KEY=your_api_key
```

---

## Example Use Cases

### Business Intelligence Workflows
Generate analytical summaries and KPI-focused reports.

### Data Quality Validation
Evaluate dataset reliability before downstream modeling.

### Academic & Research Analysis
Perform automated profiling and exploratory analysis on research datasets.

### AI-Assisted Reporting
Convert raw statistical outputs into executive-readable insights.

---

## Planned Improvements

- Predictive modeling pipelines
- Forecasting and classification support
- Real-time data ingestion
- Multi-LLM backend routing
- PostgreSQL integration
- Authentication and user sessions
- Scheduled report generation

---

## Developer

### Sujanya Srinivas

Data Science & AI Developer focused on:
- Applied Analytics Systems
- AI-Assisted Data Interpretation
- Automated Reporting Pipelines
- Data Quality Engineering
- Full-Stack AI Applications

---

## License

MIT License
