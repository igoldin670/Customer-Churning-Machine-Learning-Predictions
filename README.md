# 📊 Customer Churn Prediction Model

A complete end-to-end **machine learning churn prediction system** in a single Python file.  
This project is designed for quick deployment, education, and real-world churn modeling with clean and reproducible code.

---

## 🔥 Features

✅ Single-file ML pipeline (`churn_model.py`)  
✅ Embedded dataset included (no external data needed to train)  
✅ Automatic preprocessing (missing values, categorical encoding, type fixing)  
✅ Logistic Regression with class balancing for fair churn prediction  
✅ Saves reusable model (`churn_model.joblib`)  
✅ Predict churn on new customers via CLI  
✅ Outputs churn probability + prediction  

---

## 🧠 Model Workflow

The pipeline performs:

1. **Load & clean data**
2. **Convert Yes/No to binary**
3. **Preprocess numerical + categorical features**
4. **Fit Logistic Regression model**
5. **Evaluate model using:**
   - Accuracy
   - Precision, Recall, F1 score
   - ROC-AUC
6. **Export model for reuse**
7. **Predict churn for new datasets**

---

## 🗂 Project Structure

```
📁 Customer-Churn-ML
│
├── churn_model.py          # Main model script
├── README.md               # Project documentation (this file)
├── churn_model.joblib      # Saved model after training (generated)
└── predictions.csv         # Output for new data predictions (generated)
```

---

## 🚀 Getting Started

### 1️⃣ Install Requirements

```bash
pip install pandas scikit-learn numpy joblib
```

### 2️⃣ Train the Model

Runs training using the embedded dataset and saves `churn_model.joblib`.

```bash
python churn_model.py --train
```

### 3️⃣ Predict Churn for New Customers

Upload a CSV with the same columns (Churn column optional):

```bash
python churn_model.py --predict new_data.csv > predictions.csv
```

---

## 📤 Output Example

| ID  | churn_probability | churn_pred |
|-----|-------------------|-------------|
| C015 | 0.82              | 1           |
| C088 | 0.12              | 0           |
| C041 | 0.34              | 0           |

- `churn_pred = 1` → Model predicts customer will churn  
- `churn_pred = 0` → Customer likely to stay  

---

## 🧪 Model Evaluation Example

After training, you will see something like:

```
=== Metrics ===
  accuracy: 0.84
 precision: 0.79
    recall: 0.81
        f1: 0.80
   roc_auc: 0.87
```

---

## 🧬 Tech Stack

| Component | Purpose |
|-----------|----------|
| Python | Main language |
| scikit-learn | ML model + preprocessing |
| pandas | Data cleaning and processing |
| joblib | Save and load model |
| NumPy | Numerical operations |

---

## 🚧 Future Enhancements

Potential upgrades:

- Add RandomForest and XGBoost models for benchmarking
- Add SHAP explainability visuals
- Build a Streamlit UI for interactive predictions  
- Deploy as a REST API using FastAPI or Flask  
- Add automated retraining pipeline  

---

## 🤝 Contributing

Pull requests are welcome.  
For major changes, open an issue first to discuss what you would like to modify.

---

## 📜 License

This project is open-source and free to use for education or commercial purpose.

---

### ⭐ If this project helps you, give it a star on GitHub!

