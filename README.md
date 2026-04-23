# Clinical Comorbidity & Treatment Patterns Dashboard

A data-mining coursework project that applies **FP-Growth association rule mining** to discover hidden patterns in clinical comorbidities, presented through a professional, interactive Streamlit dashboard.

**🔗 [Live Dashboard](https://clinical-comorbidity-dashboard.streamlit.app)**  
**📄 [Full Project Report](docs/Clinical_Comorbidity_Dashboard_Report.pdf)**  
**📊 [Source Code](app.py)**

---

## Project Overview

This system demonstrates the complete end-to-end data-mining workflow: from raw clinical transaction data through algorithm selection, rule generation, filtering, interactive visualization, and cloud deployment.

### Key Features

- **FP-Growth Mining**: Discovers ~300+ association rules from 2,440 clinical visit transactions
- **Interactive Filtering**: Filter rules by primary/secondary diagnosis in real-time
- **Multi-Modal Interface**: Dashboard, Appointments, Schedule, Labs, Demographics, and Advisory panels
- **Glass-Morphism Design**: Modern, translucent UI with professional clinical aesthetics
- **One-Click Deployment**: Automatic GitHub → Streamlit Cloud deployment pipeline

---

## System Architecture

```
transactions.csv (raw data)
     ↓
TransactionEncoder (one-hot encoding)
     ↓
FP-Growth (min_support=0.01)
     ↓
Association Rules (min_confidence=0.50)
     ↓
Streamlit Dashboard (filtering, visualization, modals)
     ↓
Live at clinical-comorbidity-dashboard.streamlit.app
```

---

## Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Language** | Python 3 | Primary development language |
| **Web Framework** | [Streamlit](https://streamlit.io) | Interactive web app without frontend code |
| **Data Processing** | [pandas](https://pandas.pydata.org) | DataFrame manipulation and filtering |
| **Data Mining** | [mlxtend](http://rasbt.github.io/mlxtend/) | FP-Growth + association rule mining |
| **Visualization** | [matplotlib](https://matplotlib.org) | Heatmap colormaps |
| **Version Control** | [Git/GitHub](https://github.com/Alan-911/clinical-comorbidity-dashboard) | Repository and deployment trigger |
| **Deployment** | [Streamlit Cloud](https://streamlit.io/cloud) | Zero-config auto-deploy from GitHub |
| **Documentation** | [fpdf2](https://py-pdf.github.io/fpdf2/) | Programmatic PDF report generation |

---

## Data & Metrics

- **Dataset**: 2,440 clinical visit transactions
- **Conditions**: ~30 unique medical conditions per visit
- **Rules Generated**: 300+ association rules
- **Min Support**: 1% (pattern appears in ≥1% of visits)
- **Min Confidence**: 50% (rule is correct ≥50% of the time)
- **Ranking Metric**: Lift (descending) - how much stronger the association than random

### Sample Rule

```
Diabetes → Heart Disease
  Support:    12% (appears in 12% of visits)
  Confidence: 60% (60% of diabetics have heart disease)
  Lift:       4.0 (diabetics are 4× more likely to have heart disease)
```

---

## File Structure

```
clinical-comorbidity-dashboard/
├── app.py                              # Main Streamlit application (~550 lines)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── transactions/
│   └── transactions.csv                # Raw input: visit conditions
├── data_processed/
│   └── association_rules.csv           # Mined rules (cached)
├── visualizations/
│   └── anatomical_model.png            # 3D rotating background image
└── docs/
    ├── Clinical_Comorbidity_Dashboard_Report.pdf  # 19-page report
    └── generate_report.py              # PDF report generator
```

---

## Running Locally

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/Alan-911/clinical-comorbidity-dashboard.git
cd clinical-comorbidity-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will start on `http://localhost:8501`.

---

## Key Concepts

### Association Rule Mining
Finds rules of the form **X → Y** where:
- **X** (antecedent) = conditions a patient has
- **Y** (consequent) = conditions that frequently co-occur
- **Support** = how common the pattern is overall
- **Confidence** = how reliable the implication is (P(Y|X))
- **Lift** = strength of association vs. random chance

### Why FP-Growth?
- **Speed**: Avoids Apriori's exponential candidate generation
- **Scalability**: Uses a compressed prefix-tree (FP-tree)
- **Dense Data**: Excels on clinical transactions with many items
- **Real-time**: ~0.4s mining on 2,440 visits = responsive UI

### Why Data Mining Here?
1. **Scale**: 2,440 visits → impossible to manually spot patterns
2. **Objectivity**: Quantified metrics replace subjective judgment
3. **Discovery**: Finds unexpected correlations research may overlook
4. **Reproducibility**: Same data + same algorithm = same rules

---

## Dashboard Features

### Top Navigation
- **Dashboard**: System overview, active filters
- **Appointments**: Upcoming visits linked to mined patterns
- **Schedule**: Today's timeline with live status updates
- **Labs**: Test results with trends and clinical notes
- **Demographics**: Age-stratified prevalence (Young/Adult/Senior)

### Main Content
- **Dynamic Care Plan**: Top 3 rules for selected primary diagnosis
- **Pattern Matrix**: Top 5 rules in heatmap format (color-coded support/confidence)
- **SVG Graph Hybrid**: Interactive network view of condition relationships
- **Vital Signs**: Summary statistics (total rules, matching rules, max lift, avg confidence)

### Design Language
- **Glass-Morphism**: Translucent cards with backdrop blur
- **Grid Background**: Subtle anatomical paper effect
- **Color Coding**: Left borders indicate semantic meaning (age cohorts, priority, metrics)
- **Dark Headers**: All modals share consistent dark-gradient banners for visual cohesion

---

## Deployment

### GitHub → Streamlit Cloud Pipeline

```bash
# 1. Make changes locally
# 2. Commit and push
git add .
git commit -m "Feature: Add new modal"
git push origin main

# 3. Streamlit Cloud auto-rebuilds (~1-2 minutes)
# 4. Live at https://clinical-comorbidity-dashboard.streamlit.app
```

**No Docker, no CI configuration, no server management required.**

---

## Limitations & Disclaimers

⚠️ **Educational Project**: This is a coursework demonstration, not a clinical product.

- **Not HIPAA-compliant**: Sample data only; not suitable for real patient data
- **No regulatory review**: Makes no clinical claims; not a medical device
- **Small dataset**: 2,440 visits is adequate for learning, not for statistical inference
- **No temporality**: Rules treat each visit as an unordered set; real comorbidity is temporal
- **Substring filtering**: Uses simple string matching, not vocabulary-coded (ICD-10/SNOMED)
- **No confounder adjustment**: Rules are unadjusted for age, sex, lifestyle, genetics

---

## Future Enhancements

- **Larger dataset**: Integrate MIMIC-IV (60,000+ ICU stays) for statistical power
- **Temporal mining**: Replace FP-Growth with sequential-pattern miners (PrefixSpan, SPADE)
- **Proper vocabularies**: Use ICD-10 or SNOMED CT instead of free-text conditions
- **Supervised layer**: Use rules as features in risk-prediction models
- **Subgroup analysis**: Test rules across age/sex/geography for fairness
- **FHIR/CDS integration**: Expose as clinical decision support hook

---

## Project Report

A comprehensive **19-page PDF report** is available in [`docs/Clinical_Comorbidity_Dashboard_Report.pdf`](docs/Clinical_Comorbidity_Dashboard_Report.pdf).

**Report Contents:**
1. Executive Summary
2. Introduction: The Clinical Problem
3. Why Data Mining?
4. Association Rule Mining Theory
5. Algorithm Choice: Apriori vs FP-Growth
6. System Architecture & Data Flow
7. Technology Stack (with tool descriptions)
8. Data Pipeline & Encoding
9. Implementation Walkthrough
10. User Interface & Features
11. Deployment Strategy
12. Limitations
13. Future Work
14. Conclusion
15. References

---

## References

1. **Agrawal, R. & Srikant, R.** (1994). Fast algorithms for mining association rules in large databases. *Proceedings of VLDB '94*.

2. **Han, J., Pei, J. & Yin, Y.** (2000). Mining frequent patterns without candidate generation. *ACM SIGMOD Record*, 29(2), 1-12.

3. **Raschka, S.** (2018). MLxtend: Providing machine learning and data science utilities. *Journal of Open Source Software*, 3(24), 638.

4. **Streamlit Inc.** (2024). [Streamlit Documentation](https://streamlit.io/docs)

5. **The pandas development team.** (2024). [pandas Documentation](https://pandas.pydata.org)

6. **Hunter, J. D.** (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.

---

## Repository

- **GitHub**: [Alan-911/clinical-comorbidity-dashboard](https://github.com/Alan-911/clinical-comorbidity-dashboard)
- **Live App**: [clinical-comorbidity-dashboard.streamlit.app](https://clinical-comorbidity-dashboard.streamlit.app)
- **Created**: School data-mining coursework
- **Status**: Complete and deployed

---

*This project demonstrates the complete end-to-end workflow of turning raw data into actionable insights through association rule mining and modern web visualization.*
