# SMS Spam Detection

A Streamlit app that classifies SMS or email text as spam or not spam using a TF-IDF vectorizer and a Multinomial Naïve Bayes classifier.

## 🚀 What this project does

- Provides a friendly Streamlit UI for entering SMS/email text
- Trains a simple spam detection model at startup using sample message data
- Uses TF-IDF feature extraction and Naïve Bayes classification
- Displays prediction labels, confidence, and message details

## 📁 Project structure

- `app.py` — Streamlit application and training logic
- `vectorizer.pkl` — TF-IDF vectorizer saved object
- `model.pkl` — trained classifier saved object
- `spam.csv` — dataset file included in the repository
- `requirements.txt` — required Python packages

## ✅ Features

- Clean, attractive Streamlit UI
- Sidebar usage guide and sample messages
- Spam / Ham prediction result display
- Confidence score shown for each prediction
- In-app model details

## 💻 Installation

1. Clone or download the repository
2. Navigate to the project folder

```bash
cd /Users/satyamkumar/Desktop/SMS-Spam_Project
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL provided in the terminal.

## 🧠 Notes

- The current app trains the vectorizer and classifier from sample messages on startup, so the model remains reproducible and always fitted.
- `model.pkl` and `vectorizer.pkl` are saved artifacts for reuse and future extension.

## 📌 Want to improve it?

- Replace the sample training data with the full `spam.csv` dataset
- Add model evaluation metrics (precision, recall, F1-score)
- Support file upload for batch prediction
- Add a custom theme or dark mode

## 📦 Requirements

- Python 3.8+
- Streamlit
- scikit-learn

---

Created for SMS/Email spam classification with Streamlit.