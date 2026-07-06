import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# loading the dataset
raw_df = pd.read_csv(Path)

def clean_missing(raw_df):
    df = raw_df.copy()
    numerical_col = raw_df.select_dtypes(include="number").columns.to_list()
    categorical_col = raw_df.select_dtypes(exclude="number").columns.to_list()

    # numerical loop
    for col in numerical_col:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # categorical loop
    for col in categorical_col:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    return df