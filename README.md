# AI Resume Screening System

## 📌 Overview

AI Resume Screening System is a Machine Learning and Natural Language Processing (NLP) based web application that automatically classifies resumes into relevant job categories. The system analyzes resume content, extracts textual features using TF-IDF Vectorization, and predicts the most suitable career domain using a Logistic Regression classifier.

The project helps automate the initial resume screening process and demonstrates the practical application of Machine Learning in recruitment and talent acquisition.

---

## 🚀 Features

* Resume text analysis and classification
* Multiple job category prediction
* Confidence score generation
* NLP-based feature extraction using TF-IDF
* User-friendly Flask web interface
* Real-time prediction results

---

## 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3

### Backend

* Python
* Flask

### Machine Learning

* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression

### Data Processing

* Pandas
* NumPy

---

## 📂 Project Structure

AI_RESUME_SCREENING_SYSTEM/

├── app.py

├── train_model.py

├── resume_dataset.csv

├── model.pkl

├── static/

│ └── style.css

├── templates/

│ ├── index.html

│ └── result.html

└── requirements.txt

---

## ⚙️ Working Flow

1. User enters resume text.
2. TF-IDF Vectorizer converts text into numerical features.
3. Logistic Regression model analyzes the extracted features.
4. The system predicts the most suitable job category.
5. Confidence score is displayed along with the prediction.

---

## 📊 Supported Categories

* Data Science
* Web Developer
* Java Developer
* Android Developer
* DevOps Engineer
* Cyber Security
* UI/UX Designer

---

## 🎯 Machine Learning Pipeline

Resume Text
↓
TF-IDF Vectorization
↓
Feature Extraction
↓
Logistic Regression Classifier
↓
Job Category Prediction
↓
Confidence Score

---

## 🔧 Installation

1. Clone the repository

git clone <repository-url>

2. Install dependencies

pip install -r requirements.txt

3. Train the model

python train_model.py

4. Run the application

python app.py

5. Open browser

http://127.0.0.1:5000

---

## 📈 Future Enhancements

* PDF Resume Upload
* Resume Parsing
* Skill Extraction
* Job Recommendation Engine
* Deep Learning Models (BERT)
* Dashboard Analytics
* Cloud Deployment

---

## 👨‍💻 Author

Afroj Shaikh

Computer Science & Engineering Student

Passionate about Machine Learning, Web Development, and Artificial Intelligence.
