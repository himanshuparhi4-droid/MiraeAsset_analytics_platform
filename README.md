# Mirae Asset Digital Platform - User Analytics

[![Live Dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://miraeassetanalytics.streamlit.app/)

> [Launch Live Dashboard](https://miraeassetanalytics.streamlit.app/)

A reproducible analytics system built on a synthetic fintech dataset. The project covers the full analytics lifecycle across 9 Jupyter notebooks and 27 phases, from valid raw data generation through predictive modelling, business simulation, and Streamlit deployment.

---

## Project Summary

Current metrics are generated from `data/processed/project_metrics.json` after running `python scripts/pipeline.py`.

| Metric | Value |
|--------|-------|
| Total users | 10,000 registered users from 12,500 visitors |
| Total revenue (6 months) | Rs 38.10M |
| Churn rate | 17.4% |
| Conversion rate | 48.0% |
| Avg LTV per buyer | Rs 7,937 |
| Churn model AUC | 0.826 |
| Best paid CAC | Rs 207 per buyer (Referral) |
| Best paid ROAS | 38.0x (Referral) |

---

## 27-Phase Roadmap

### Stage 1 - Foundation
| Phase | Description | Notebook |
|-------|-------------|----------|
| 1 | Data Generation - 5 synthetic raw CSVs with visitor funnel and valid post-signup behaviour | NB01 |
| 2 | Data Cleaning - null handling, type validation, temporal logic | NB02 |
| 3 | Feature Engineering - engagement score, tenure-aware churn flag, 27-column registered-user master dataset | NB02 |
| 4 | Exploratory Data Analysis - distributions, revenue by state/channel/device | NB03 |

### Stage 2 - Core Analytics
| Phase | Description | Notebook |
|-------|-------------|----------|
| 5 | Funnel Analysis - visitor-to-signup-to-purchase conversion, drop-off rates, funnel by channel and device | NB04 |
| 6 | Advanced Churn Analysis - churn profile, timing, pre-churn signals, at-risk segmentation | NB04 |
| 7 | Cohort Analysis - session retention heatmap, cumulative LTV curves, repeat purchase by cohort | NB04 |
| 8 | Marketing Channel Analysis - volume, conversion rate, churn rate, engagement by channel | NB05 |
| 9 | CAC & ROI - cost allocation, CAC per channel, ROAS, LTV:CAC ratio, payback period | NB05 |

### Stage 3 - Business Intelligence
| Phase | Description | Notebook |
|-------|-------------|----------|
| 10 | Revenue Segmentation - by channel, device, geography, payment method, Pareto analysis | NB05 |
| 11 | Time-Based Analysis - monthly trend, MoM growth, day-of-week pattern | NB05 |
| 12 | User Segmentation - RFM scoring, rule-based segments, K-Means clustering for buyers (K=4) plus Non-Buyer label | NB06 |
| 13 | Funnel Segmentation - funnel conversion by channel and device with drop-off breakdown | NB04 |

### Stage 4 - Advanced Analytics
| Phase | Description | Notebook |
|-------|-------------|----------|
| 14 | Predictive Modelling - Logistic Regression vs Random Forest vs tuned GBM; AUC 0.826 | NB07 |
| 15 | Explainability - built-in feature importance, permutation importance, partial dependence | NB07 |
| 16 | A/B Testing - hypothesis design, sample size calculation, Z-test & Chi-square | NB07 |
| 17 | Pricing Strategy - AOV distribution, revenue concentration, optimisation scenarios | NB08 |
| 18 | Business Simulation - 4 independent scenarios plus 1 combined scenario | NB08 |

### Stage 5 - Strategic Decision Intelligence
| Phase | Description | Notebook |
|-------|-------------|----------|
| 19 | KPI Tree - revenue decomposition, sensitivity analysis by lever | NB08 |
| 20 | Root Cause Analysis - monthly revenue decomposition and calendar-normalised checks | NB08 |
| 21 | Decision Simulation - impact vs effort matrix, ranked action plan | NB08 |
| 22 | Data Pipeline Architecture - system overview, data flow diagram | NB09 |

### Stage 6 - Production & Deployment
| Phase | Description | Notebook / File |
|-------|-------------|-----------------|
| 23 | Power BI Dashboard | Single-file dashboard: `powerbi/Mirae_Asset_Analytics.pbix` |
| 24 | Automation - `pipeline.py` rebuilds master data, segmentation, marketing summary, and project metrics | NB09 + `scripts/pipeline.py` |
| 25 | Deployment - 6-tab Streamlit dashboard with dynamic metrics, segmentation, and drilldown views | NB09 + `app/app.py` |
| 26 | Documentation - README, requirements.txt, .gitignore | NB09 |
| 27 | Video Walkthrough | Planned - Loom recording link to be added |

---

## Key Findings

1. **Referral is the most efficient paid channel in the current baseline.** Referral has the lowest paid CAC per buyer at about Rs 207 and the highest paid ROAS at about 38.0x.
2. **Revenue concentration is moderate.** The top 10% of all users generate about 38.1% of revenue, while the top 10% of buyers generate about 21.2%.
3. **Churn remains predictable without leakage.** The tuned Gradient Boosting churn model reaches AUC 0.826 after excluding direct recency leakage, indirect recency proxies, and duplicate RFM monetary/frequency columns.
4. **AOV improvement is the largest broad lever.** A 10% AOV lift implies about Rs 3.81M gross revenue upside on the current baseline.
5. **The funnel is causal and has real top-of-funnel drop-off.** Notebook 01 now generates 12,500 visitors, 10,000 signups, 7,000 cart users, and 4,800 purchasers in chronological order.
6. **The production pipeline is complete.** One command rebuilds processed data, marketing metrics, out-of-fold churn risk scores, and model artifacts.

---

## Project Structure

```text
Mirae Asset major/
|
+-- data/
|   +-- raw/                  # Source CSVs generated by NB01
|   |   +-- users.csv
|   |   +-- sessions.csv
|   |   +-- transactions.csv
|   |   +-- events.csv
|   |   +-- campaigns.csv
|   +-- processed/            # Output of scripts/pipeline.py
|       +-- user_data.csv            # Master 27-column registered-user dataset
|       +-- user_data_segmented.csv  # RFM segments, K-Means labels, OOF churn scores
|       +-- marketing_summary.csv    # CAC, ROAS, LTV:CAC, payback by channel
|       +-- project_metrics.json     # Headline metrics for README/dashboard
|
+-- models/
|   +-- model_artifacts.pkl          # Single bundle: churn model, metadata, scalers, K-Means, labels
|
+-- notebooks/
|   +-- 01_data_generation.ipynb
|   +-- 02_data_cleaning_feature_engineering.ipynb
|   +-- 03_EDA_Insights_fixed.ipynb
|   +-- 04_funnel_churn_cohort_analysis.ipynb
|   +-- 05_marketing_cac_revenue.ipynb
|   +-- 06_user_segmentation_clustering.ipynb
|   +-- 07_predictive_modelling_ab_testing.ipynb
|   +-- 08_business_strategy_simulation.ipynb
|   +-- 09_pipeline_deployment.ipynb
|
+-- scripts/
|   +-- pipeline.py           # Automated end-to-end data processing
|
+-- app/
|   +-- app.py                # Streamlit dashboard
|
+-- powerbi/
|   +-- Mirae_Asset_Analytics.pbix  # Single-file Power BI dashboard for submission
|
+-- requirements.txt
+-- .gitignore
+-- README.md
```

The Power BI dashboard is stored as a `.pbix` file for a clean submission. The
PBIX was built from processed project outputs, including
`data/processed/user_data_segmented.csv`, `marketing_summary.csv`,
`project_metrics.json`, and the raw events table for the funnel.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Regenerate processed data
python scripts/pipeline.py

# 3. Launch dashboard
streamlit run app/app.py
```

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine learning | scikit-learn (GBM, Random Forest, Logistic Regression, K-Means) |
| Statistics | scipy (Z-test, Chi-square, Mann-Whitney U) |
| Dashboard | Streamlit |
| Environment | Jupyter Notebook |

---

## Skills Demonstrated

End-to-end data pipeline, temporal data validation, RFM segmentation, K-Means clustering, GBM churn prediction, permutation feature importance, A/B test design and analysis, business simulation, KPI tree, root cause analysis, Streamlit deployment, and automated metric generation.
