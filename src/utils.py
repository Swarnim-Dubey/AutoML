def get_columns_type(df):
    numerical_col = df.select_dtypes(include="number").columns.to_list()
    categorical_col = df.select_dtypes(exclude="number").columns.to_list()
    return numerical_col, categorical_col