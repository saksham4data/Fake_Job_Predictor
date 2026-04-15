# 🕵️ Fake Job Predictor

Fake Job Predictor is a machine learning-based application that detects fraudulent job postings by analyzing their descriptions. It uses Natural Language Processing (TF-IDF) and a Logistic Regression model to identify scam patterns and provide predictions along with a risk score.

---

## 🚀 Features

* 🔍 Detects whether a job posting is **Real or Fraudulent**
* 🧠 Uses NLP (TF-IDF) for text feature extraction
* ⚖️ Handles imbalanced data using `class_weight='balanced'`
* 📊 Provides **risk/confidence score**
* 💻 Interactive **Streamlit UI**
* 🛡️ Includes error handling for robust performance

---

## 🏗️ Project Structure

```
Fake_Job_Predictor/
│
├── Data/
│   └── cleaned_data.csv
│
├── Trained_Models/
│   ├── fake_job_model_*.pkl
│   ├── tfidf_vectorizer_*.pkl
│
├── Dependencies/
│   ├── requirements.txt
│
├── streamlit_app.py
├── train_model.py
├── requirements.txt
└── README.md
└── LICENSE
```

---

## ⚙️ How It Works

1. **Data Preprocessing**

   * Handles missing values
   * Combines relevant text fields
   * Cleans and normalizes text

2. **Feature Engineering**

   * Converts text into numerical form using **TF-IDF**

3. **Model Training**

   * Logistic Regression model trained on labeled dataset
   * Handles imbalance using class weighting

4. **Prediction**

   * User inputs job description
   * Text is cleaned → vectorized → passed to model
   * Output: Real / Fraud + probability score

---

## 🧪 Model Performance

* High recall for fraudulent jobs (important for scam detection)
* Balanced trade-off between precision and recall
* Evaluated using:

  * Precision
  * Recall
  * F1 Score
  * ROC-AUC

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone https://github.com/saksham4data/Fake_Job_Predictor.git
cd Fake_Job_Predictor
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```
streamlit run streamlit_app.py
```

---

## 🧾 Input Format

Paste any job description like:

```
"We are hiring remote workers. No experience required. Earn money from home..."
```

---

## 📌 Output

* ✅ Real Job (Confidence Score)
* ⚠️ Fraudulent Job (Risk Score)

---

## ⚠️ Limitations

* Model depends on text patterns → may misclassify well-written scams
* Performance depends on training data quality
* Does not use external company verification

---

## 🔮 Future Improvements

* Add explainability (highlight suspicious words)
* Use advanced NLP models (BERT, transformers)
* Deploy as web service
* Integrate real-time job validation APIs

---

## 👨‍💻 Author

**Saksham Nagar**

---

## 📜 License

This project is open-source and available under the MIT License.
