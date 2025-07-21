

<img src = "static/pos.png" alt="Prediction Example" width="500" height="500"/>


# 🧠 Sentiment Analysis using Scikit-learn

This project is a machine learning-based **sentiment and emotion classifier** trained on Twitter data. It uses **Scikit-learn's Multinomial Naive Bayes** model to classify tweets into four categories:
**Positive**, **Negative**, **Neutral**, and **Irrelevant**.

The trained model is saved in the file `senti.ykl` and can be reused for predictions on new text data.

---

## 📂 Project Structure

```
machine_learning_model_sentiment_analysis/
│
├--- model_creations.py # python file to create train model and save it using joblib
│--- senti.ykl        # Saved trained model
├── main.py          # Main file for server with fastapi 
├── templates/
│   ├── index.html   # The frontend page for UI
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

## 🧪 Model Details

* **Algorithm**: Multinomial Naive Bayes
* **Library**: Scikit-learn
* **Features**: TF-IDF vectors
* **Dataset**: [Kaggle Twitter Sentiment Dataset](https://www.kaggle.com/datasets)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/machine_learning_model_sentiment_analysis.git
cd machine_learning_model_sentiment_analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Load and Use the Model

You can use the `senti.ykl` file to predict sentiment on new text data without retraining:

```python
# predict.py (example usage)
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
model = joblib.load('models/senti.ykl')

# Example text
text = ["I love this product!", "This is the worst movie ever."]

# Load your saved vectorizer or recreate it as used in training
vectorizer = joblib.load('models/vectorizer.ykl')  # If saved
X = vectorizer.transform(text)

# Predict
predictions = model.predict(X)
print(predictions)
```

---

## 📈 Performance Metrics

Model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

These metrics can be reproduced by running the training notebook or script.

---

## 📌 Use Cases

* Twitter sentiment monitoring
* Social media content filtering
* Customer feedback analysis
* Detecting irrelevant or off-topic posts

---

## 📎 License

This project is open-source and available under the [MIT License](LICENSE).

---
