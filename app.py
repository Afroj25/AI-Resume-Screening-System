from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model pipeline
# Make sure you run train_model.py first to generate model.pkl
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get resume text from textarea
    resume_text = request.form.get("resume", "").strip()

    # Validate input
    if not resume_text:
        return render_template(
            "result.html",
            category="No resume text provided.",
            confidence=0
        )

    # Predict category
    prediction = model.predict([resume_text])[0]

    # Predict probabilities (if supported by the classifier)
    try:
        probabilities = model.predict_proba([resume_text])[0]
        confidence = max(probabilities) * 100
    except Exception:
        confidence = 0

    return render_template(
        "result.html",
        category=prediction,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)