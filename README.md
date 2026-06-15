# 🛡️ ShieldUPI
## AI-Based UPI Fraud Detection System using Machine Learning

An end-to-end Machine Learning application that analyzes UPI transaction details and predicts whether a transaction is **Valid** or **Fraudulent**. The project combines data preprocessing, feature engineering, model training, and a user-friendly web interface for real-time fraud detection.

---

## 🚀 Features

- 🔍 Real-time UPI Fraud Detection
- 🤖 Random Forest Machine Learning Model
- 📊 Interactive Analytics Dashboard
- 📈 Model Performance Comparison
- 🧠 Feature Engineering & One-Hot Encoding
- 💾 Transaction History Storage
- 🔐 User Authentication System
- ⚡ Probability-based Risk Scoring

---

## 🏗️ Project Architecture

```
User Input
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
One-Hot Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Random Forest Model
      │
      ▼
Prediction
      │
      ▼
Fraud Probability
      │
      ▼
Final Result
```

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Random Forest Classifier

### Data Processing

- Pandas
- NumPy

### Backend

- Flask

### Database

- SQLite

### Model Storage

- Joblib

### Frontend

- HTML
- CSS
- JavaScript

### Version Control

- Git
- GitHub

---

## 📂 Project Structure

```
upi-fraud-detection/

│── app.py

│── requirements.txt

│── generate_dataset.py

│── generate_notebook.py

│── train_rf_model.py

│── model_building.ipynb

│── rf_model.pkl

│── scaler.pkl

│── expected_columns.pkl

│── templates/

│── static/

│── scratch/
```

---

## 📊 Model Information

| Algorithm | Status |
|-----------|----------|
| Logistic Regression | Tested |
| K-Nearest Neighbors | Tested |
| Support Vector Machine | Tested |
| Decision Tree | Tested |
| ✅ Random Forest | Deployed |

---

## 📈 Workflow

1. User enters transaction details
2. Data is preprocessed
3. Categorical values are encoded
4. Features are scaled
5. Random Forest model performs prediction
6. Fraud probability is calculated
7. Final decision is displayed

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/labeebzaidi/upi-fraud-detection.git
```

Move into project folder

```bash
cd upi-fraud-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---


## 🎯 Future Improvements

- Live Banking API Integration
- Deep Learning-based Fraud Detection
- Explainable AI (SHAP)
- Real-time Monitoring Dashboard
- Streamlit Version
- Cloud Deployment
- Email/SMS Fraud Alerts

---

## 👨‍💻 Author

**Mohd Labeeb Zaidi**

- LinkedIn: https://linkedin.com/in/labeebzaidi
- GitHub: https://github.com/labeebzaidi

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
