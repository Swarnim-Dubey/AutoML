import numpy as np
import pandas as pd
from pathlib import Path
from utils import get_columns_type

# loading the dataset
def load_dataset(path):
    return pd.read_csv(path)

# fixing the dtypes
def fix_dtypes(df):
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], "coerce")
        if converted.notna().sum() / len(df) > 0.9:
            df[col] = converted
    return df

def clean_missing(raw_df):
    df = raw_df.copy()
    df = fix_dtypes(df)
    numerical_col, categorical_col = get_columns_type(df)

    # numerical loop
    for col in numerical_col:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # categorical loop
    for col in categorical_col:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    return df