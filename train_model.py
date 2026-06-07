import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("resume_dataset.csv")

# Features and labels
X = df["Resume"]
y = df["Category"]

# Split dataset into training and testing sets
# stratify=y keeps category balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a pipeline:
# 1. Convert text into TF-IDF vectors
# 2. Train a Logistic Regression classifier
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model trained successfully!")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the complete pipeline as model.pkl
# (This includes both TF-IDF vectorizer and classifier)
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Saved model to model.pkl")