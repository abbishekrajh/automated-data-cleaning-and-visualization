import streamlit as st
import pandas as pd
import plotly.express as px
from modules.ml_engine import apply_knn_imputer, detect_anomalies_isolation_forest
from modules.data_cleaning import remove_duplicates, handle_missing_values
from modules.data_visualization import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_categorical_counts,
)

st.set_page_config(page_title="Automated Data Cleaner", layout="wide")
st.title("Automated Data Cleaning & Profiling Dashboard")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

with st.sidebar:
        st.header("ML Engine Settings")
        contamination = st.slider(
            "Expected Contamination Rate",
            min_value=0.01,
            max_value=0.40,
            value=0.05,
            step=0.01,
            help="Estimated proportion of outliers in the dataset (5% default)."
        )
        knn_k = st.slider(
            "KNN Imputer Neighbors (k)",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Number of nearest neighbors used to impute missing numeric values."
        )
if uploaded_file is not None:
    
  
    if "df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.session_state.file_name = uploaded_file.name

   
    current_df = st.session_state.df

   
    st.write("### Data Quality Overview")
    total_rows = len(current_df)
    total_cols = current_df.shape[1]
    duplicate_count = current_df.duplicated().sum()
    null_count = current_df.isnull().sum().sum()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Rows", total_rows)
    m_col2.metric("Total Columns", total_cols)
    m_col3.metric("Duplicate Rows", duplicate_count)
    m_col4.metric("Total Missing Values", null_count)

    st.write("---")
    st.write("### Clean Your Data")

    col_clean1, col_clean2 = st.columns(2)

  
    with col_clean1:
        st.write("#### 1. Deduplication")
        if st.button("Remove Duplicate Rows"):
            dups_found = current_df[current_df.duplicated()]
            num_dups = len(dups_found)
            
            if num_dups > 0:
                # Update persistent state
                st.session_state.df = remove_duplicates(current_df)
                st.success(f"Removed {num_dups} duplicate row(s)!")
                with st.expander("View Removed Duplicate Rows"):
                    st.dataframe(dups_found)
                st.rerun()  
            else:
                st.info("No duplicates to remove.")

  
    with col_clean2:
        st.write("#### 2. Handle Missing Values")
        strategy_choice = st.selectbox(
            "Select strategy:",
            ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"]
        )
        if st.button("Apply Strategy"):
            st.session_state.df = handle_missing_values(current_df, strategy=strategy_choice)
            st.success(f"Applied: {strategy_choice}")
            st.rerun()  

    st.write("---")
    st.write("### Data Preview")
    st.dataframe(st.session_state.df)

    st.write("---")
    st.write("### Interactive Visualizations")

    viz_choice = st.selectbox(
        "Select visualization type:",
        ["Numeric Distribution", "Correlation Heatmap", "Categorical Frequency"]
    )

    numeric_columns = st.session_state.df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = st.session_state.df.select_dtypes(include=["object"]).columns.tolist()

    if viz_choice == "Numeric Distribution":
        if numeric_columns:
            selected_num = st.selectbox("Select numeric column:", numeric_columns)
            fig = plot_distribution(st.session_state.df, selected_num)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No numeric columns available.")

    elif viz_choice == "Correlation Heatmap":
        if len(numeric_columns) >= 2:
            fig = plot_correlation_heatmap(st.session_state.df)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")

    elif viz_choice == "Categorical Frequency":
        if categorical_columns:
            selected_cat = st.selectbox("Select categorical column:", categorical_columns)
            fig = plot_categorical_counts(st.session_state.df, selected_cat)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No categorical columns available.")


    st.write("---")
    st.write("### Export Processed Data")
    csv_data = st.session_state.df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned CSV",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    st.write("---")
    st.write("### 🤖 Multivariate Anomaly Detection")

    # Step 1: Impute missing values with KNN
    df_clean = apply_knn_imputer(current_df, n_neighbors=knn_k)

    # Step 2: Score and flag anomalies with Isolation Forest
    df_scored, anomaly_count = detect_anomalies_isolation_forest(df_clean, contamination=contamination)

    # Step 3: Anomaly Metrics Cards
    total_records = len(df_scored)
    anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0.0

    ano_col1, ano_col2, ano_col3 = st.columns(3)
    ano_col1.metric("Total Records Evaluated", f"{total_records:,}")
    ano_col2.metric("Flagged Anomalies", f"{anomaly_count:,}")
    ano_col3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

    # Step 4: Scatter Plot Feature Space
    numeric_cols = [c for c in df_scored.select_dtypes(include=['number']).columns if c != "anomaly_score"]

    if len(numeric_cols) >= 2:
        st.write("#### Feature Space Explorer")
        axis1, axis2 = st.columns(2)
        with axis1:
            x_axis = st.selectbox("X-Axis Feature", numeric_cols, index=0)
        with axis2:
            y_axis = st.selectbox("Y-Axis Feature", numeric_cols, index=1)

        fig = px.scatter(
            df_scored,
            x=x_axis,
            y=y_axis,
            color="is_anomaly",
            color_discrete_map={True: "#EF553B", False: "#636EFA"},
            hover_data=["anomaly_score"],
            title=f"{y_axis} vs. {x_axis} (Colored by Anomaly Status)",
            template="plotly_white"
        )
        fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Upload data with at least 2 numeric features to display the scatter plot.")

    # Step 5: Audit Table & Download
    st.write("#### Flagged Records Audit")
    outliers_df = df_scored[df_scored["is_anomaly"]].sort_values(by="anomaly_score", ascending=True)

    if not outliers_df.empty:
        st.dataframe(outliers_df, width="stretch")
        csv_bytes = outliers_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Flagged Outliers (CSV)",
            data=csv_bytes,
            file_name="flagged_anomalies.csv",
            mime="text/csv"
        )
    else:
        st.success("No anomalies detected at the current sensitivity setting.")