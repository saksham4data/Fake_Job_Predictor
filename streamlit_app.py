"""
Fake Job Predictor Application
A Streamlit app that predicts fraudulent jobs using a trained LogisticRegression Model.

Author - Saksham Nagar

"""

import streamlit as st
import re
import pickle
from pathlib import Path 

#Load Model and TF-IDF

@st.cache_resource
def load_models():
    try:
        model_path = Path("Trained_Models")

        if not model_path.exists():
            st.error(f"Model directory not found: {model_path}")
            st.stop()

        model_files = sorted(model_path.glob("fake_job_model_*.pkl"))
        tfidf_files = sorted(model_path.glob("tfidf_vectorizer_*.pkl"))

        if not model_files or not tfidf_files:
            st.error("No model or TF-IDF files found in the directory")
            st.stop()
        
        latest_model = model_files[-1]
        latest_tfidf = tfidf_files[-1]

        with open(latest_model, 'rb') as f:
            model = pickle.load(f)
        
        with open(latest_tfidf, 'rb') as f:
            tfidf = pickle.load(f)
        
        return model, tfidf
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

#Text Cleaning Function

def clean_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]'," ", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return ""

#Predict Function

def predict(text, model, tfidf):
    try:
        cleaned = clean_text(text)

        if not cleaned:
            raise ValueError("Input text is empty after cleaning")

        vectorized = tfidf.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0][1]

        return prediction, prob
    except Exception as e:
        st.error(f"Error in prediction{str(e)}")
        return None, None

#UI

st.set_page_config(page_title="👮🏼Fake Job Predictor")
st.title("👮🏼 Fake Job Predictor")
st.write("Paste a job description below to check if it's real or fraudulent.")

#Load model
model, tfidf = load_models()

if model is None or tfidf is None:
    st.stop()

#Input
user_input = st.text_area("Enter Job description", height=200)

#Predict Button
if st.button("Analyze Job"):

    if not user_input.strip():
        st.warning("Please enter a job description.")
    else:
        word_count = len(user_input.split())

        if word_count < 30 or len(user_input) < 100:
             st.warning("⚠️ Please provide a proper job description (minimum 30 words)")
             st.stop()
        prediction, prob = predict(user_input, model, tfidf)

        if prediction is not None:
            st.subheader("Result:")

            if prediction==1:
                st.error(f"Fraudulent Job Risk: {prob:.2f}")

            else:
                st.success(f"Real Job (Confidence: {1-prob:.2f})")