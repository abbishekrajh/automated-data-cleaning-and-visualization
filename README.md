# Automated Data Cleaning, Profiling & Anomaly Detection Dashboard

An interactive Streamlit web application for automated data quality profiling, rule-based cleaning, exploratory visualization, and multivariate machine learning anomaly detection.

## Key Features

- **Data Ingestion & Session Management:** Upload `.csv` or `.xlsx` datasets with persistent state tracking across UI interactions.
- **Automated Data Quality Profiling:** Instant overview of dataset dimensions, duplicate record counts, and missing value distributions.
- **Interactive Data Cleaning:** One-click deduplication and missing value handling.
- **Exploratory Data Visualization:** Interactive Plotly charts including marginal distribution histograms, rounded numeric correlation heatmaps, and top-N categorical bar charts.
- **Hybrid ML Engine (`modules/ml_engine.py`):**
  - **KNN Imputation:** Estimates and fills missing numerical values using Scikit-Learn `KNNImputer` with configurable k-neighbors (`2-10`) while preserving DataFrame index alignment.
  - **Multivariate Anomaly Detection:** Identifies multi-dimensional outliers using Scikit-Learn `IsolationForest` (`random_state=42`) with adjustable contamination sensitivity (`1%-20%`), defensive shape guards, continuous anomaly decision scoring, an interactive 2D feature space scatter explorer, and one-click CSV export of flagged anomalies.

## Project Structure

- `app.py`: Main Streamlit dashboard and UI layout
- `modules/data_cleaning.py`: Deduplication and data cleaning utilities
- `modules/data_profiling.py`: Summary statistics and data quality checks
- `modules/data_visualization.py`: Plotly Express chart builders
- `modules/ml_engine.py`: KNNImputer and IsolationForest pipeline

## Tech Stack

- **Frontend / App Framework:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`KNNImputer`, `IsolationForest`)
- **Visualization:** Plotly Express

## Getting Started

1. Clone the repository:
   `git clone https://github.com/abbishekrajh/automated-data-cleaning-and-visualization.git`
2. Install dependencies:
   `pip install streamlit pandas numpy scikit-learn plotly openpyxl`
3. Launch the app:
   `streamlit run app.py`
