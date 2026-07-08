from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
from utils import get_columns_type

# defining the models that are going to be used in the pipeline
models = {
    "classification": {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(),
        "XGBClassifier": XGBClassifier()
    },
    "regression": {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "XGBRegressor": XGBRegressor()
    }
}

def build_preprocessing(num_col, cat_col):
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_col),
        ("cat", categorical_transformer, cat_col)
    ])
    return preprocessor

# looping through the models and creating a pipeline for each one
def build_model_pipelines(X, y, problem_type):
    num_col, cat_col = get_columns_type(X)
    preprocessor = build_preprocessing(num_col, cat_col)
    
    pipelines = {}
    for model_name, model in models[problem_type].items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipelines[model_name] = pipeline
    return pipelines


# selecting the best model based on the problem type
def select_best_model(X, y, problem_type):
    pipelines = build_model_pipelines(X, y, problem_type)
    if problem_type == "classification":
        scoring = "accuracy"
    else:
        scoring = "r2"

    results = {}
    for model_name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
        results[model_name] = scores.mean()
    
    best_model_name = max(results, key=results.get)
    best_pipeline = pipelines[best_model_name]
    best_pipeline.fit(X, y)
    return best_model_name, best_pipeline

def save_artifact(artifact, path):
    joblib.dump(artifact, path)
    print(f"Artifact saved at: {path}")