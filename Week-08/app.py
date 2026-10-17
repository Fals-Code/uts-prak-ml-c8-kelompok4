"""UI Streamlit untuk prediksi Attrition dengan model terbaik Week 08."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.comparison import comparison_frame, run_experiment
from src.data_pipeline import transform_for_prediction

WEEK_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Prediksi Employee Attrition", page_icon="??", layout="wide")


@st.cache_resource
def load_experiment():
    return run_experiment()


experiment = load_experiment()
pipeline = experiment["pipeline"]
best_model = experiment["best_model"]
best_name = experiment["best_model_name"]
metrics = experiment["metrics"]

st.title("Prediksi Employee Attrition")
st.caption("Week 08 � Praktikum Machine Learning � Kelompok 4 C8")

best_accuracy = metrics[best_name]["accuracy"]
col1, col2, col3 = st.columns(3)
col1.metric("Model UI", best_name)
col2.metric("Accuracy testing", f"{best_accuracy:.2%}")
col3.metric("Split data", "80% / 20%")

st.info(
    "Model UI dipilih otomatis dari accuracy tertinggi pada split yang sama. "
    "Input harus mengikuti kolom data hasil preprocessing UTS."
)

tab_sample, tab_upload, tab_result = st.tabs(["Contoh data testing", "Upload CSV", "Hasil evaluasi"])

with tab_sample:
    sample_number = st.number_input(
        "Nomor sampel testing",
        min_value=1,
        max_value=len(pipeline.X_test_original),
        value=1,
        step=1,
    )
    sample_position = int(sample_number) - 1
    sample = pipeline.X_test_original.iloc[[sample_position]]
    actual = int(pipeline.y_test.iloc[sample_position])
    st.dataframe(sample, use_container_width=True, height=210)
    if st.button("Prediksi sampel", type="primary"):
        transformed = transform_for_prediction(sample, pipeline)
        prediction = int(best_model.predict(transformed)[0])
        label = "Yes" if prediction == 1 else "No"
        actual_label = "Yes" if actual == 1 else "No"
        st.success(f"Prediksi Attrition: {label}")
        st.write(f"Nilai aktual pada data testing: **{actual_label}**")

with tab_upload:
    uploaded = st.file_uploader("Pilih CSV hasil preprocessing UTS", type=["csv"])
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        st.dataframe(input_df.head(), use_container_width=True)
        try:
            transformed = transform_for_prediction(input_df, pipeline)
            predictions = best_model.predict(transformed)
            result = input_df.copy()
            result["Prediksi_Attrition"] = ["Yes" if value == 1 else "No" for value in predictions]
            st.success(f"Berhasil memprediksi {len(result)} baris.")
            st.dataframe(result, use_container_width=True)
        except ValueError as error:
            st.error(str(error))

with tab_result:
    table = comparison_frame(metrics).copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.write("Confusion Matrix model UI:", metrics[best_name]["confusion_matrix"])


