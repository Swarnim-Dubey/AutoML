import streamlit as st
import joblib
import pandas as pd
from src.detector import detect_id_columns, detect_problem_type, drop_id_columns, encode_target, profile_columns, split_features_target
from src.eda import clean_missing
from src.pipeline import select_best_model, save_artifact

st.title("AutoML")
st.write("This is an automated ML model that can be used on a raw dataset")
tab1, tab2 = st.tabs(["Train", "Predict"])

with tab1:
    file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if file is not None:
        df_preview = pd.read_csv(file)
        target_col = st.selectbox("Select the target column", df_preview.columns)

        if st.button("Train Model"):
            df = clean_missing(df_preview)
            id_cols = detect_id_columns(df)
            df = drop_id_columns(df, id_cols)

            X, y = split_features_target(df, target_col=target_col)

            problem_type = detect_problem_type(y)
            encoder = None
            if problem_type == "classification":
                y, encoder = encode_target(y)
            num_col, cat_col = profile_columns(X)
            cat_values = {}
            for col in cat_col:
                cat_values[col] = X[col].unique().tolist()

            best_model_name, best_pipeline, results = select_best_model(X, y, problem_type)

            artifact = {
                "pipeline": best_pipeline,
                "problem_type": problem_type,
                "num_col": num_col,
                "cat_col": cat_col,
                "cat_values": cat_values,
                "encoder": encoder
            }
            save_artifact(artifact, "models/best_model.pkl")
            st.success(f"Training complete. Best model: {best_model_name}")
            st.bar_chart(results)

with tab2:
    artifact = joblib.load("models/best_model.pkl")
    pipeline = artifact["pipeline"]
    num_col = artifact["num_col"]
    cat_col = artifact["cat_col"]
    cat_values = artifact["cat_values"]
    encoder = artifact["encoder"]
    problem_type = artifact["problem_type"]

    user_input = {}
    for col in num_col:
        user_input[col] = st.number_input(f"Enter value for {col}")
    for col in cat_col:
        user_input[col] = st.selectbox(f"Select {col}", cat_values[col])

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = pipeline.predict(input_df)

        if problem_type == "classification":
            result = encoder.inverse_transform(prediction)[0]
        else:
            result = prediction[0]
        st.success(f"Prediction: {result}")