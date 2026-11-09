import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

def apply_knn_imputer(df: pd.DataFrame, n_neighbors: int = 5):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include = ['number']).columns
    if len(numeric_cols) == 0 or not df_clean[numeric_cols].isnull().any().any():
        return df_clean
    imputer = KNNImputer(n_neighbors = n_neighbors)
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols]) 
    return df_clean

def detect_anomalies_isolation_forest(df: pd.DataFrame, contamination: float = 0.05):
    df_scored = df.copy()
    numeric_df = df_scored.select_dtypes(include = ['number']).dropna()
    if numeric_df.shape[0] < 10 or numeric_df.shape[1] < 2:
        df_scored['is_anomaly'] = False
        df_scored['anomaly_score'] = 0
        return df_scored, 0
    
    
    iso = IsolationForest(
        n_estimators = 100,
        contamination = contamination,
        random_state = 42
    )
    predictions = iso.fit_predict(numeric_df)
    scores = iso.decision_function(numeric_df)

    df_scored["is_anomaly"] = False
    df_scored["anomaly_score"] = 0.0

    df_scored.loc[numeric_df.index, "is_anomaly"] = (predictions == -1)
    df_scored.loc[numeric_df.index, "anomaly_score"] = scores

    anomaly_count = int(df_scored["is_anomaly"].sum())
    return df_scored, anomaly_count

    
