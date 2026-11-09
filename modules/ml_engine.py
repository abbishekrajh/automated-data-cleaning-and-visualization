import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

def apply_knn_imputer(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    df_clean = df.copy()