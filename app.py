from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
DATA_PATH = ROOT / "data" / "engine_data.csv"

st.set_page_config(page_title="Predictive Maintenance Demo", page_icon="⚙️", layout="centered")
st.title("Engine Condition Prediction")
st.caption("A machine-learning demonstration using engine sensor readings.")
st.info(
    "The dataset target is encoded as 0/1, but its operational meaning has not been independently confirmed. "
    "Treat the output as a recorded model class, not a confirmed failure or maintenance alert."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run the ML pipeline or add models/best_model.joblib to the repository.")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_summary():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found at data/engine_data.csv.")
    df = pd.read_csv(DATA_PATH)
    features = [c for c in df.columns if c != "Engine Condition"]
    return df[features].median(numeric_only=True), df[features].min(), df[features].max(), features

try:
    model = load_model()
    medians, minima, maxima, feature_names = load_feature_summary()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Enter sensor readings")
with st.form("sensor_inputs"):
    values = {}
    columns = st.columns(2)
    for i, feature in enumerate(feature_names):
        low = float(minima[feature])
        high = float(maxima[feature])
        default = float(medians[feature])
        # Number inputs allow values outside the observed sample range when needed.
        with columns[i % 2]:
            values[feature] = st.number_input(
                feature,
                value=default,
                min_value=min(0.0, low) if low >= 0 else None,
                max_value=None,
                format="%.3f",
                help=f"Dataset median: {default:.3f}; observed range: {low:.3f} to {high:.3f}.",
            )
    submitted = st.form_submit_button("Predict engine condition", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=feature_names)
    prediction = int(model.predict(input_df)[0])
    st.subheader("Model output")
    st.metric("Predicted recorded class", str(prediction))
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        probability_df = pd.DataFrame({"Recorded class": model.classes_, "Model probability": probabilities})
        st.dataframe(probability_df, hide_index=True, use_container_width=True)
    st.caption("This is a model prediction on the dataset's encoded target. Confirm label definitions and validate operational thresholds before using it for maintenance decisions.")
