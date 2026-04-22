# 🫀 CardioScan — Heart Disease Prediction System

CardioScan is an **end-to-end machine learning project** that predicts the **likelihood of heart disease using patient clinical data.**

The project demonstrates the **complete ML workflow — from data preprocessing and model training to evaluation and deployment through an interactive Streamlit application.**

It focuses on building a practical system for **risk prediction while showcasing core machine learning concepts.**

---

## Project Overview

CardioScan demonstrates the complete **ML lifecycle**:
- **Data preprocessing and feature preparation**  
- **Training multiple classification models**
- **Evaluating models using standard metrics**
- **Selecting the best-performing model**  
- **Deploying the model via a Streamlit web application**  

---

##  Key Highlights

- Built a **modular ML pipeline**
- **Trained multiple models:**
  - Random Forest
  - Logistic Regression
- **Selected best model using F1-score**
- **Implemented full evaluation:**
  - Accuracy, Precision, Recall, F1, ROC-AUC
- Developed an interactive **Streamlit dashboard**
  
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

### 2. Model Training
**- Pipeline:**
  - **StandardScaler + RandomForest**
  - **StandardScaler + LogisticRegression**

### 3. Model Evaluation
**- Metrics:**
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **ROC-AUC**

### 4. Model Performance

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------------------|----------|----------|--------|----------|--------|
| Random Forest          | 83.6%    | 0.84     | 0.84   | 0.84     | 0.92   |
| Logistic Regression    | 85.2%    | 0.87     | 0.84   | 0.86     | 0.93   |

### 5. Model Selection
- Best model selected using **F1-score**
- **Saved as `model.pkl`**

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
streamlit run app.py
```
--- 

## Application Features
- **Interactive patient input panel**  
- **Real-time health metrics visualization**  
- **ML-based heart disease risk prediction**  
- **Probability-based risk scoring**  
- **Clean and intuitive Streamlit UI**

## Note
This project is inspired by a machine learning course.
The implementation, structure, and enhancements are done independently for learning and portfolio purposes.

