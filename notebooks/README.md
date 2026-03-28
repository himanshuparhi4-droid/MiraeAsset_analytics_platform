# Mirae Asset Digital Platform — User Analytics

A production-grade analytics system built on a synthetic fintech dataset.
Covers the complete analytics lifecycle across 9 Jupyter notebooks and 27 phases —
from raw data generation through predictive modelling, business simulation, and Streamlit deployment.

---

## Project Summary

| Metric | Value |
|--------|-------|
| Total users | 10,000 |
| Total revenue (6 months) | Rs 37.6M |
| Churn rate | 47.6% |
| Conversion rate | 47.7% |
| Avg LTV per buyer | Rs 7,889 |
| Churn model AUC | 0.84 |
| Best paid CAC | Rs 104 (Google Ads) |

---

## 🚀 27-Phase Roadmap

### 🟢 Stage 1 — Data Foundation
| Phase | Task | Notebook |
|------|------|----------|
| 1 | Data Generation | NB01 |
| 2 | Data Cleaning | NB02 |
| 3 | Feature Engineering | NB02 |
| 4 | Exploratory Analysis | NB03 |

### 🔵 Stage 2 — Core Analytics
| Phase | Task | Notebook |
|------|------|----------|
| 5 | Funnel Analysis | NB04 |
| 6 | Churn Analysis | NB04 |
| 7 | Cohort Analysis | NB04 |
| 8 | Marketing Analysis | NB05 |
| 9 | CAC & ROI | NB05 |

### 🟣 Stage 3 — Business Intelligence
| Phase | Task | Notebook |
|------|------|----------|
| 10 | Revenue Segmentation | NB05 |
| 11 | Time Analysis | NB05 |
| 12 | User Segmentation (Clustering) | NB06 |
| 13 | Funnel Segmentation | NB04 |

### 🟠 Stage 4 — Advanced Analytics
| Phase | Task | Notebook |
|------|------|----------|
| 14 | Predictive Modeling | NB07 |
| 15 | Explainability | NB07 |
| 16 | A/B Testing | NB07 |
| 17 | Pricing Strategy | NB08 |
| 18 | Business Simulation | NB08 |

### 🔴 Stage 5 — Strategic Decision Intelligence 🔥
| Phase | Task | Notebook |
|------|------|----------|
| 19 | KPI Tree | NB08 |
| 20 | Root Cause Analysis | NB08 |
| 21 | Decision Simulation | NB08 |
| 22 | Data Pipeline Architecture | NB09 |

### 🟢 Stage 6 — Production & Deployment 🚀
| Phase | Task | File |
|------|------|------|
| 23 | Power BI Dashboard | Planned |
| 24 | Automation Pipeline | pipeline.py |
| 25 | Streamlit Deployment | app/app.py |
| 26 | Documentation | README |
| 27 | Video Walkthrough | Planned |

---

## 📌 Key Business Insights

1. Google Ads delivers highest ROI — CAC Rs 104 vs Rs 241.
2. Top 10% buyers generate 43% of revenue.
3. Churn predictable at AUC 0.84.
4. AOV +10% increases revenue significantly.
5. February dip is seasonal, not structural.

---

## Quick Start

pip install -r requirements.txt  
python scripts/pipeline.py  
streamlit run app/app.py  

---

## Tech Stack

- pandas, numpy  
- matplotlib, seaborn, plotly  
- scikit-learn  
- scipy  
- streamlit  

