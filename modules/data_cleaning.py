import pandas as pd

def remove_duplicates(df):
   return df.drop_duplicates().reset_index(drop=True) 
   
def handle_missing_values(df, strategy="Drop Rows"):
    df_clean = df.copy()

    if strategy == "Drop Rows":
       return df_clean.dropna().reset_index(drop=True)

    elif strategy == "Fill with Mean":
       numeric_cols = df_clean.select_dtypes(include=['number']).columns
       df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
       return df_clean

    elif strategy == "Fill with Median":
         numeric_cols = df_clean.select_dtypes(include=['number']).columns
         df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
         return df_clean

    elif strategy == "Fill with Mode":
            cat_cols = df_clean.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                if not df_clean[col].mode().empty:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            return df_clean