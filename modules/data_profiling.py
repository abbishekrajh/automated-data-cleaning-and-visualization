import pandas as pd
def get_profile_summary(df):
    total_rows, total_columns = df.shape
    duplicates = df.duplicated().sum()
    missing_values = df.isnull().sum().sum()
    return {
        "Total Rows": total_rows,
        "Total Columns": total_columns,
        "Duplicate Rows": duplicates,
        "Missing Values": missing_values
    }
def get_column_summary(df):
    summary_df = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique(),
    })
    return summary_df