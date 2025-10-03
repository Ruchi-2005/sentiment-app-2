# streamlit_topic_sentiment_app.py

import streamlit as st
import pandas as pd
import joblib
import os
import json
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Custom transformer (same as used in training)
# -------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [x.lower().strip() for x in X]

# -------------------------
# Load model and metrics
# -------------------------
model_file = os.path.join(os.path.dirname(__file__), "topic_sentiment_model.joblib")
metrics_file = os.path.join(os.path.dirname(__file__), "topic_metrics.json")
dataset_file = os.path.join(os.path.dirname(__file__), "topic_sentiment_dataset.csv")

if not os.path.exists(model_file):
    st.error("⚠️ Model file not found. Please run train_model.py first.")
    st.stop()

# Load trained model
try:
    model = joblib.load(model_file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load dataset for topics
if os.path.exists(dataset_file):
    df_topics = pd.read_csv(dataset_file)
    topics = df_topics['topic'].unique().tolist()
else:
    topics = []

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Topic Sentiment Analyzer", page_icon="📝", layout="wide")
st.title("📝 Topic-Based Sentiment Analyzer")
st.info("Select a topic, enter text about it, and see sentiment predictions.")

# -------------------------
# Sidebar: select topic
# -------------------------
topic_selected = st.selectbox("Select a Topic:", topics)

# -------------------------
# Main input area
# -------------------------
st.subheader(f"Enter your sentences about '{topic_selected}':")
user_input = st.text_area("Enter one or multiple sentences (separate each by a new line)")

if st.button("Predict Sentiment"):
    if user_input.strip():
        sentences = [s for s in user_input.split("\n") if s.strip()]
        predictions = model.predict(sentences)

        # Display predictions
        st.subheader("Predictions per sentence:")
        for sentence, pred in zip(sentences, predictions):
            if pred.lower() == "positive":
                st.success(f"✅ {sentence} → {pred}")
            elif pred.lower() == "negative":
                st.error(f"❌ {sentence} → {pred}")
            else:
                st.info(f"ℹ️ {sentence} → {pred}")

        # Aggregate overall sentiment
        counts = pd.Series(predictions).value_counts()
        overall = counts.idxmax()
        st.subheader("Overall Sentiment for this Topic:")
        if overall.lower() == "positive":
            st.success(f"Overall Sentiment: {overall}")
        elif overall.lower() == "negative":
            st.error(f"Overall Sentiment: {overall}")
        else:
            st.info(f"Overall Sentiment: {overall}")
    else:
        st.warning("Please enter at least one sentence.")
