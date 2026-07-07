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