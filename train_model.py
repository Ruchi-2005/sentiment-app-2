# train_model_improved.py

import pandas as pd
import joblib
import json
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Custom text transformer
# -------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    """Clean text for preprocessing"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [x.lower().strip() for x in X]

# -------------------------
# Load dataset
# -------------------------
dataset_file = "topic_sentiment_dataset.csv"
if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"Dataset file '{dataset_file}' not found. Please create it first.")

df = pd.read_csv(dataset_file)
df.dropna(subset=['text', 'label'], inplace=True)

# -------------------------
# Optional: Expand dataset for short sentence variety
# -------------------------
def augment_sentences(row):
    text = row['text']
    label = row['label']
    variants = []

    # Simple short sentence variations
    if len(text.split()) <= 3:
        if label == "positive":
            variants.extend([text + "!", "Absolutely " + text, "I really " + text])
        elif label == "negative":
            variants.extend([text + ".", "I don't like it", "Not " + text])
        elif label == "neutral":
            variants.extend([text + ".", "Just " + text, text])

    return variants

aug_texts, aug_labels = [], []
for _, row in df.iterrows():
    variants = augment_sentences(row)
    for v in variants:
        aug_texts.append(v)
        aug_labels.append(row['label'])

# Combine original + augmented
X = pd.concat([df['text'], pd.Series(aug_texts)], ignore_index=True)
y = pd.concat([df['label'], pd.Series(aug_labels)], ignore_index=True)

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Build pipeline with TF-IDF
# -------------------------
pipeline = Pipeline([
    ('cleaner', TextCleaner()),
    ('vectorizer', TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ('classifier', LogisticRegression(max_iter=500))
])

# -------------------------
# Train model
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate model
# -------------------------
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
labels = list(df['label'].unique())

metrics = {
    "accuracy": accuracy,
    "report": report,
    "confusion_matrix": cm.tolist(),
    "labels": labels
}

# -------------------------
# Save model and metrics
# -------------------------
model_file = "topic_sentiment_model.joblib"
metrics_file = "topic_metrics.json"

joblib.dump(pipeline, model_file)
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"✅ Model saved to {model_file}")
print(f"✅ Metrics saved to {metrics_file}")
print(f"✅ Training complete! Accuracy: {accuracy:.2f}")
