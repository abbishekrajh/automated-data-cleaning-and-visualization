import streamlit as st
import pandas as pd
from modules.data_cleaning import remove_duplicates, handle_missing_values
from modules.data_visualization import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_categorical_counts,
)

st.set_page_config(page_title="Automated Data Cleaner", layout="wide")
st.title("Automated Data Cleaning & Profiling Dashboard")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

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
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available.")

    elif viz_choice == "Correlation Heatmap":
        if len(numeric_columns) >= 2:
            fig = plot_correlation_heatmap(st.session_state.df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")

    elif viz_choice == "Categorical Frequency":
        if categorical_columns:
            selected_cat = st.selectbox("Select categorical column:", categorical_columns)
            fig = plot_categorical_counts(st.session_state.df, selected_cat)
            st.plotly_chart(fig, use_container_width=True)
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