# 🗂️ Customer Complaint Auto-Classifier — End-to-End NLP Pipeline

> Automatically classify consumer complaints into 5 business sectors using Machine Learning and Natural Language Processing.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project implements a **complete end-to-end NLP pipeline** to automatically classify customer complaints into one of 5 real-world business sectors:

| # | Category |
|---|----------|
| 1 | 🏦 Banking & Finance |
| 2 | 📱 Telecom & Internet |
| 3 | 🛒 E-commerce & Shopping |
| 4 | 🍔 Food Delivery & Restaurant |
| 5 | 🏥 Healthcare & Pharmacy |

Built as part of the **NAVTTC AI/ML/DL Course** at **DevCastle-IUB-BWP**.  

---

## 🎯 Problem Statement

Companies receive thousands of complaints daily. Manually routing each complaint to the right department is slow and error-prone. This pipeline **automatically classifies** incoming complaints — enabling faster resolution, reduced mis-routing, and better customer satisfaction scores.

---

## 📁 Repository Structure

---

## 🔬 Pipeline Steps

```
Raw Text Data
     │
     ▼
① Problem Selection  →  Customer Complaint Classification (5 classes)
     │
     ▼
② Data Collection    →  800+ samples (Trustpilot, ConsumerAffairs, Google Reviews)
     │
     ▼
③ EDA                →  Class distribution, text length, word count plots
     │
     ▼
④ Preprocessing      →  Lowercase → Remove URLs/Punctuation →
                         Tokenize → Remove Stopwords → Lemmatize
     │
     ▼
⑤ Feature Extraction →  Bag of Words  &  TF-IDF  (ngram 1-2, 5000 features)
     │
     ▼
⑥ Model Training     →  Logistic Regression | Decision Tree |
                         SVM (LinearSVC) | Random Forest
     │
     ▼
⑦ Evaluation         →  Accuracy | Precision | Recall | F1 | Confusion Matrix
     │
     ▼
⑧ Visualization      →  10 plots via Matplotlib & Seaborn
     │
     ▼
⑨ Hyperparameter     →  GridSearchCV (5-fold CV) on SVM
   Tuning
     │
     ▼
⑩ Final Comparison   →  All models ranked by F1-Score
     │
     ▼
⑪ Vectorizer Analysis→  BoW vs TF-IDF — why TF-IDF wins
     │
     ▼
⑫ Conclusion         →  Real-world deployment plan + live prediction demo
```

---

## 🏆 Results

| Model | Vectorizer | Accuracy | F1-Score |
|-------|-----------|----------|----------|
| **SVM (LinearSVC) — Tuned** | **TF-IDF** | **~97%** | **~97%** |
| SVM (LinearSVC) | TF-IDF | ~96% | ~96% |
| Logistic Regression | TF-IDF | ~95% | ~95% |
| Random Forest | TF-IDF | ~93% | ~93% |
| Decision Tree | TF-IDF | ~85% | ~85% |
| All models | BoW | lower | lower |

> ✅ **Best Model: SVM (LinearSVC) + TF-IDF (GridSearchCV Tuned)**

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/majidhussain-ai/customer-complaint-nlp.git
cd customer-complaint-nlp
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python nlp_pipeline.py
```

### 4. Live Prediction (after running)
```python
import joblib

model = joblib.load('complaint_classifier.pkl')

complaints = [
    "My credit card was charged twice and the bank is ignoring me.",
    "Internet has been down for 3 days, no technician arrived.",
    "I ordered shoes but received a completely broken item.",
]

for c in complaints:
    print(f"→ {model.predict([c])[0]}")
```

**Output:**
```
→ Banking & Finance
→ Telecom & Internet
→ E-commerce & Shopping
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
wordcloud
joblib
```

Install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud joblib
```

---

## 🌍 Real-World Applications

- 🏦 **Banking** — Auto-route fraud, loan, and account complaints to correct teams
- 📱 **Telecom** — Instantly separate network outages from billing issues
- 🛒 **E-commerce** — Flag high-priority delivery failures and fraud complaints
- 🍔 **Food Delivery** — Immediately escalate food-safety and tampering reports
- 🏥 **Healthcare** — Route medication errors to pharmacovigilance departments

---

## 🚀 Deployment Plan

```
Model (pkl)  →  FastAPI Endpoint  →  POST /classify
                     │
                     ▼
             { "category": "Banking & Finance" }
                     │
                     ▼
          CRM Integration (Zendesk / Freshdesk / Salesforce)
                     │
                     ▼
          Auto-ticket routing to correct department
```

---

## 📝 Why TF-IDF > Bag of Words?

| Feature | BoW | TF-IDF |
|---------|-----|--------|
| Term weighting | Raw count | Weighted by rarity |
| Noise suppression | ❌ Treats all words equally | ✅ Down-weights common words |
| Discriminative signal | Weak | Strong |
| Model performance | Lower | Higher on all 4 models |

---

## 👤 Author

**[Majid Hussain]**  
NAVTTC AI/ML/DL Batch  
DevCastle-IUB-BWP  
📧 majid.hussain.ml.eng@gmail.com 
🔗 [LinkedIn](https://linkedin.com/in/majidhussain-ai)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and build upon it.

---

<p align="center">
  Made with ❤️ for NAVTTC AI/ML/DL Course | DevCastle-IUB-BWP
</p>
```
