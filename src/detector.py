import numpy as np
import pandas as pd
from utils import get_columns_type

def split_features_target(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def detect_problem_type(y):
    if y.dtype == "object":
        return "classification"
    unique_ratio = y.nunique() / len(y)

    if unique_ratio < 0.05:
        return "classification"
    else:
        return "regression"
    
def profile_columns(X):
    numerical_col, categorical_col = get_columns_type(X)
    return numerical_col, categorical_col

def detect_id_columns(df, threshold=0.95):
    id_cols = []
    for col in df.columns:
        ratio = df[col].nunique() / len(df)
        if ratio > threshold:
            id_cols.append(col)
    return id_cols

def drop_id_columns(df, id_cols):
    return df.drop(columns=id_cols)