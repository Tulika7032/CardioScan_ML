# 🫀 CardioScan — Heart Disease Prediction System

CardioScan is an **end-to-end machine learning project** that predicts the **likelihood of heart disease using patient clinical data.**

The project demonstrates the **complete ML workflow — from data preprocessing and model optimization to evaluation and through an interactive Streamlit application.**

It focuses on building a practical system for **risk prediction while showcasing core machine learning concepts.**

---

## Features

- End-to-end machine learning workflow
- Automated data preprocessing
- Hyperparameter tuning using GridSearchCV
- Multiple classification models
- Comprehensive model evaluation
- Automatic best model selection
- Interactive Streamlit dashboard
- Probability-based risk prediction
- Feature importance visualization
- Modular project structure

---


## Tech Stack

**Languages & Libraries**
- **Python**
- **NumPy, Pandas**
- **Scikit-learn**
- **Joblib**

**Frontend / App**
- **Streamlit**

---

## Machine Learning Workflow

### 1. Data Preprocessing
- **Load dataset**
- **Split into features and target**
- **Train-test split (80/20)**
- **Standardization using `StandardScaler`**

### 2. Model Training
**- Pipeline:**
  - **StandardScaler + RandomForest**
  - **StandardScaler + LogisticRegression**
  - Hyperparameter tuning using **GridSearchCV**

### 3. Model Evaluation
**- Metrics:**
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **ROC-AUC**
  - **Confusion Matrix**
  - **Classification Report**

### 4. Model Performance

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------------------|----------|----------|--------|----------|--------|
| Random Forest          | 83.6%    | 0.84     | 0.84   | 0.84     | 0.92   |
| Logistic Regression    | 85.2%    | 0.87     | 0.84   | 0.86     | 0.93   |

### 5. Model Selection
- Best model selected using **F1-score**
- **Saved as `model.pkl`**
- Generated model comparison and evaluation reports

### 6. Prediction
- **Load saved model**
- **Predict on new input data**

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heart-disease-project.git
cd heart-disease-project
```

### 2. Create a Virtual Enviornment
```bash
python -m venv venv
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python main.py
```

### 5. Run the streamlit app
```bash
streamlit run app/app.py
```
--- 

## Application Features
- **Interactive patient input panel**  
- **Real-time health metrics visualization**  
- **ML-based heart disease risk prediction**  
- **Probability-based risk scoring**  

## Note
This project is inspired by a machine learning course.
The implementation, structure, and enhancements are done independently for learning and portfolio purposes.

