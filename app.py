from pathlib import Path

import joblib
import streamlit as st

# Page setup
st.set_page_config(
    page_title="PhishShield AI",
    page_icon="🛡️",
    layout="centered"
)

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# Application header
st.title("🛡️ PhishShield AI")
st.subheader("Intelligent Phishing Email Detection")

st.write(
    "Analyse email text using Natural Language Processing "
    "and Machine Learning."
)

st.divider()

# User input
email_text = st.text_area(
    "📧 Paste an email to analyse:",
    height=220,
    placeholder="Enter the email subject and message here..."
)

# Prediction
if st.button("🔍 Analyse Email", use_container_width=True):

    if not email_text.strip():
        st.warning("Please enter an email first.")

    else:
        prediction = int(model.predict([email_text])[0])

        # Probability of phishing class (label 1)
        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.named_steps["classifier"].classes_)
        phishing_probability = float(
            probabilities[classes.index(1)]
        )

        st.divider()

        if prediction == 1:
            st.error("🚨 Potential Phishing Email Detected")
            st.write(
                "The model classified this email as phishing. "
                "Avoid clicking suspicious links or sharing "
                "sensitive information."
            )
        else:
            st.success("✅ Email Classified as Safe")
            st.write(
                "The model classified this email as safe, "
                "but this does not guarantee the email is trustworthy."
            )

        st.metric(
            "Model phishing probability",
            f"{phishing_probability * 100:.1f}%"
        )

        st.progress(phishing_probability)

        st.caption(
            "This probability is a model estimate, not a "
            "calibrated security risk score."
        )

st.divider()

st.caption(
    "Built with Python, Scikit-learn, TF-IDF, "
    "Logistic Regression and Streamlit."
)
st.caption(
    "Educational ML project. Do not rely on this tool "
    "alone for security decisions."
)