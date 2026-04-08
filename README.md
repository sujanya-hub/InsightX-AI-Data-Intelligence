# InsightX: Modular AI Data Intelligence Platform

InsightX is a production-grade analytics platform designed to automate the end-to-end data analysis lifecycle. It integrates statistical processing, data quality engineering, and Large Language Models (LLMs) to convert raw datasets into structured insights and executive-ready reports.

[**Live Demo**](https://insightx-ai-data-intelligence.onrender.com) | [**Source Code**](https://github.com/sujanya-hub/InsightX-AI-Data-Intelligence)

---

## Strategic Overview

InsightX simulates a real-world data analytics workflow within a single system. It enables users to move from raw data to decision-ready insights through an automated pipeline:

1. **Ingestion & Validation** — Load and validate dataset structure
2. **Quality Engineering** — Quantify data health (missing values, duplicates, outliers)
3. **Automated Preprocessing** — Clean, normalize, and prepare data
4. **Analytical Processing** — Generate KPIs, trends, and EDA outputs
5. **AI Interpretation** — Translate statistical findings into natural language insights
6. **Report Generation** — Produce structured, shareable PDF reports

---

## Core Modules

### Data Engineering & Quality Assurance

* Automated statistical profiling (distributions, cardinality, summary statistics)
* Data quality scoring based on completeness, uniqueness, and anomalies
* Preprocessing pipelines for imputation, outlier handling, and normalization

### Analytics & AI Copilot

* Exploratory Data Analysis (EDA) with visual summaries
* Scenario-based decision analysis for identifying risks and opportunities
* LLM-powered explanations using Groq (Llama / Mixtral)
* Executive summaries for non-technical stakeholders

### Reporting & Visualization

* Dashboard-style visualizations using Matplotlib and Seaborn
* Automated PDF report generation using ReportLab
* Structured outputs suitable for business and presentation use

---

## Technical Highlights

* Designed a **modular, decoupled architecture** separating data processing, analytics, AI, and UI layers
* Implemented a **data quality scoring system** combining multiple heuristics into a single metric
* Integrated **LLM-based reasoning (Groq API)** for real-time natural language insight generation
* Built **end-to-end pipeline automation** from ingestion → analysis → reporting
* Developed a **state-managed Streamlit interface** with multi-module routing and caching
* Optimized for deployment with **environment-based configuration (local + cloud compatibility)**

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
│   └── reporting/        # PDF generation (ReportLab)
│
├── modules/              # Streamlit UI and routing
├── utils/                # Shared utilities
└── app.py                # Application controller
```

---

## Technical Stack

| Layer           | Technology                 |
| --------------- | -------------------------- |
| Language        | Python 3.9+                |
| Framework       | Streamlit                  |
| Data Processing | Pandas, NumPy              |
| Analytics       | Scikit-learn               |
| LLM Integration | Groq API (Llama / Mixtral) |
| Visualization   | Matplotlib, Seaborn        |
| Reporting       | ReportLab                  |
| Deployment      | Render                     |

---

## Installation

### Local Setup

```bash
git clone https://github.com/sujanya-hub/InsightX-AI-Data-Intelligence.git
cd InsightX-AI-Data-Intelligence
pip install -r requirements.txt
```

### Environment Configuration

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_api_key_here"
```

### Run Application

```bash
streamlit run app.py
```

---

## Deployment (Render)

* Build Command:
  `pip install -r requirements.txt`

* Start Command:
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

* Environment Variable:
  `GROQ_API_KEY=your_api_key`

---

## Use Cases

* Business intelligence and reporting
* Data quality validation workflows
* Academic and research analysis
* Decision support systems
* Portfolio demonstration of end-to-end data pipelines

---

## Roadmap

* Predictive modeling integration (classification / forecasting)
* Real-time data ingestion support
* Multi-LLM backend support
* Persistent storage layer (PostgreSQL)
* User authentication and session management

---

## Author

Sujanya Srinivas
Data Science & Analytics — Jain University

License: MIT
