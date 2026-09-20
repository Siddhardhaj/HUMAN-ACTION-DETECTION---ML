import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Human Action Detection",
    page_icon="🏃",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("human_action_model.pkl")


package = load_model()

model = package["model"]
scaler = package["scaler"]
lower_bounds = package["lower_bounds"]
upper_bounds = package["upper_bounds"]
feature_columns = package["feature_columns"]
activity_labels = package["activity_labels"]


# ============================================================
# TITLE
# ============================================================

st.title("🏃 Human Action Detection")
st.markdown(
    "### MHEALTH Sensor-Based Activity Classification"
)

st.write(
    "Enter the 12 sensor readings below to predict the "
    "corresponding human activity."
)

st.divider()


# ============================================================
# SENSOR INPUTS
# ============================================================

st.subheader("📊 Sensor Measurements")

col1, col2, col3, col4 = st.columns(4)

with col1:
    alx = st.number_input("ALX", value=0.0, format="%.4f")
    aly = st.number_input("ALY", value=0.0, format="%.4f")
    alz = st.number_input("ALZ", value=0.0, format="%.4f")

with col2:
    glx = st.number_input("GLX", value=0.0, format="%.4f")
    gly = st.number_input("GLY", value=0.0, format="%.4f")
    glz = st.number_input("GLZ", value=0.0, format="%.4f")

with col3:
    arx = st.number_input("ARX", value=0.0, format="%.4f")
    ary = st.number_input("ARY", value=0.0, format="%.4f")
    arz = st.number_input("ARZ", value=0.0, format="%.4f")

with col4:
    grx = st.number_input("GRX", value=0.0, format="%.4f")
    gry = st.number_input("GRY", value=0.0, format="%.4f")
    grz = st.number_input("GRZ", value=0.0, format="%.4f")


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Detect Human Activity", use_container_width=True):

    # Create input DataFrame using EXACT feature order
    input_data = pd.DataFrame(
        [[
            alx, aly, alz,
            glx, gly, glz,
            arx, ary, arz,
            grx, gry, grz
        ]],
        columns=feature_columns
    )

    # Apply the SAME clipping used during training
    input_clipped = input_data.clip(
        lower=lower_bounds,
        upper=upper_bounds,
        axis=1
    )

    # Apply the SAME RobustScaler
    input_scaled = scaler.transform(input_clipped)

    # Make prediction
    prediction_encoded = model.predict(input_scaled)[0]

    # Convert encoded value to activity name
    prediction_encoded = int(prediction_encoded)

    predicted_activity = activity_labels[prediction_encoded]

    # Display result
    st.success(
        f"🎯 Predicted Activity: **{predicted_activity}**"
    )

    st.info(
        f"Activity Code: {prediction_encoded}"
    )


# ============================================================
# ACTIVITY REFERENCE
# ============================================================

with st.expander("📋 View Activity Classes"):

    activity_df = pd.DataFrame(
        list(activity_labels.items()),
        columns=["Code", "Activity"]
    )

    st.table(activity_df)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Human Action Detection using the MHEALTH dataset | "
    "Machine Learning Internship Project"
)
