# 🏥 Healthcare Predictive Analytics — Diabetes Detection
### Internship Final Project — Project 5

> An end-to-end Machine Learning pipeline to predict diabetes risk in patients using clinical health records — with full EDA, ethical data handling, model comparison, and feature importance analysis.

---

## 📌 Project Overview

Diabetes affects over 500 million people worldwide and is one of the leading causes of death and disability. Early detection is critical — catching the disease before symptoms worsen can prevent serious complications like kidney failure, blindness, and heart disease.

This project builds a complete ML classification pipeline using real patient health data to **identify high-risk individuals early**, supporting clinical decision-making through data-driven insights.

| | |
|---|---|
| **Domain** | Healthcare / Medical AI |
| **Type** | Binary Classification |
| **Dataset** | Diabetes Prediction Dataset (Kaggle) |
| **Records** | 100,000 patients |
| **Target** | `diabetes` — 1 (Diabetic) / 0 (Non-Diabetic) |
| **Churn Rate** | 8.5% positive cases (imbalanced dataset) |

---

## 📂 Project Structure

```
healthcare-predictive-analysis/
│
├── diabetes_notebook.ipynb            # Main Jupyter Notebook (run this)
├── diabetes_prediction_dataset.csv    # Patient dataset (100,000 rows)
├── diabetes_model_metrics.csv         # Saved model comparison results
└── README.md                          # Project documentation
```

---

## 📊 Dataset Description

**Source:** [Diabetes Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Patient gender (Male / Female / Other) |
| `age` | Numerical | Patient age in years (0.08 – 80) |
| `hypertension` | Binary | Has hypertension — 1 Yes / 0 No |
| `heart_disease` | Binary | Has heart disease — 1 Yes / 0 No |
| `smoking_history` | Categorical | Smoking status (never / former / current / ever / not current / No Info) |
| `bmi` | Numerical | Body Mass Index |
| `HbA1c_level` | Numerical | Glycated haemoglobin level (%) — key diabetes marker |
| `blood_glucose_level` | Numerical | Fasting blood glucose level (mg/dL) |
| **`diabetes`** | **Target** | **Whether patient has diabetes (1 = Yes, 0 = No)** |

### 🩺 Clinical Reference Values

| Biomarker | Normal | Pre-Diabetic | Diabetic |
|---|---|---|---|
| HbA1c Level | < 5.7% | 5.7% – 6.4% | ≥ 6.5% |
| Blood Glucose | < 100 mg/dL | 100 – 125 mg/dL | ≥ 126 mg/dL |
| BMI | 18.5 – 24.9 | 25 – 29.9 | ≥ 30 (Obese) |

---

## ⚙️ Installation & Setup

### 1. Clone or download the project files

```bash
git clone https://github.com/yourusername/healthcare-predictive-analysis.git
cd healthcare-predictive-analysis
```

### 2. Install required libraries

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn
```

### 3. Rename the dataset file

```bash
mv 1782476950768_diabetes_prediction_dataset.csv diabetes_prediction_dataset.csv
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook diabetes_notebook.ipynb
```

### 5. Run all cells

Go to **Kernel → Restart & Run All**

> ⏱️ **Expected Runtime: 2–4 minutes** (100k rows, 4 models with cross-validation)

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis (EDA)
- Diabetes class distribution (bar + pie charts)
- Age, BMI, blood glucose distribution by diabetes status
- HbA1c and blood glucose boxplots with clinical thresholds
- Diabetes rate by gender, smoking history, and hypertension
- Scatter plots — Age vs BMI, HbA1c vs Blood Glucose
- Full feature correlation heatmap

### Step 2 — Preprocessing
- Label encoded categorical features (`gender`, `smoking_history`)
- Applied **SMOTE** to handle severe class imbalance (8.5% positive rate)
- Normalized all features using **StandardScaler**
- Stratified 80/20 train-test split to preserve class ratios

### Step 3 — Model Training
Trained 4 classification models with 5-fold Stratified Cross-Validation:

| Model | Key Parameters | Reason Chosen |
|---|---|---|
| Logistic Regression | `C=1.0`, `max_iter=1000` | Interpretable baseline |
| Random Forest | `n_estimators=100`, `max_depth=10` | Handles non-linearity well |
| XGBoost | `n_estimators=100`, `lr=0.1`, `depth=5` | Best overall performance |
| Naive Bayes | `GaussianNB` | Fast probabilistic model |

> ⚠️ **Why not SVM?** SVM with RBF kernel has O(n²) complexity — on 100,000 rows it takes 30–60 minutes. Replaced with Naive Bayes, which trains in under 2 seconds with comparable recall performance. This is a valid real-world engineering trade-off.

### Step 4 — Evaluation
- Accuracy, Recall, ROC-AUC, F1-Score on held-out test set
- ROC curves for all models on one chart
- Confusion matrices for all models
- Full classification reports (precision, recall, F1 per class)
- Feature importance analysis (Random Forest + XGBoost)

---

## 📈 Results

| Model | CV AUC | Accuracy | Recall | ROC-AUC | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | ~0.95 | ~0.86 | ~0.85 | ~0.95 | ~0.70 |
| Random Forest | ~0.97 | ~0.96 | ~0.72 | ~0.97 | ~0.80 |
| **XGBoost** | **~0.97** | **~0.97** | **~0.77** | **~0.97** | **~0.82** |
| Naive Bayes | ~0.91 | ~0.84 | ~0.83 | ~0.91 | ~0.67 |

> ✅ **XGBoost** achieved the best overall performance.
>
> ⚠️ In healthcare, **Recall is the most important metric** — a missed diabetic patient (False Negative) has serious clinical consequences. Logistic Regression and Naive Bayes trade accuracy for higher recall, which may be preferable in a real screening scenario.

---

## 🔑 Key Findings

| Feature | Clinical Insight |
|---|---|
| **HbA1c Level** | #1 predictor — values ≥ 6.5% are the strongest diabetes signal |
| **Blood Glucose** | #2 predictor — fasting glucose ≥ 126 mg/dL is a diagnostic threshold |
| **BMI** | Higher BMI significantly increases diabetes risk (especially BMI ≥ 30) |
| **Age** | Risk rises sharply after age 45 |
| **Hypertension** | Strong comorbidity — hypertensive patients show 3× higher diabetes rate |
| **Smoking History** | Former and current smokers show elevated risk vs never-smokers |
| **Gender** | Minimal difference — slight elevation in male patients |

### 💡 Clinical Recommendation
> Patients with **HbA1c ≥ 5.7%** + **blood glucose ≥ 100 mg/dL** + **BMI ≥ 28** should be prioritized for early screening and lifestyle intervention programs. This combination flags the highest-risk cohort with strong model confidence.

---

## ⚖️ Ethical Considerations

This project was built with responsible AI principles at every stage:

| Principle | Action Taken |
|---|---|
| **Patient Privacy** | Dataset is fully anonymized — no names, IDs, or PII present |
| **Fairness** | Model evaluated across gender groups to detect bias |
| **Transparency** | All predictions traceable via feature importance scores |
| **No Record Removal** | All 100,000 patient records included — no selective filtering |
| **Balanced Training** | SMOTE applied to prevent model bias toward majority class |
| **Clinical Limitation** | Model is a screening aid only — not a replacement for diagnosis |

> 🏥 **Medical Disclaimer**: This model is intended for research and early screening support only. All clinical decisions must be made by qualified healthcare professionals. Model outputs should never replace a licensed physician's assessment.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | ML models, preprocessing, metrics |
| XGBoost | Gradient boosting classifier |
| Imbalanced-learn | SMOTE oversampling for class imbalance |
| Matplotlib | Charts and visualizations |
| Seaborn | Statistical plots and heatmaps |
| Jupyter Notebook | Interactive development environment |

---

## 📉 Visualizations Included

- ✅ Diabetes class distribution (bar + pie)
- ✅ Age, BMI, blood glucose histograms by diabetes status
- ✅ HbA1c and blood glucose boxplots with clinical threshold lines
- ✅ Diabetes rate by gender, smoking history, hypertension
- ✅ Age vs BMI scatter plot (sample n=5,000)
- ✅ HbA1c vs blood glucose scatter with diagnostic thresholds
- ✅ Full correlation heatmap
- ✅ ROC curves (all 4 models)
- ✅ Confusion matrices (all 4 models)
- ✅ Metric comparison bar chart
- ✅ Feature importance (Random Forest + XGBoost)

---

## 🚀 Future Improvements

- [ ] Hyperparameter tuning with Optuna or GridSearchCV
- [ ] Add deep learning model (MLP / TabNet)
- [ ] Threshold tuning to maximize Recall for clinical use
- [ ] SHAP values for patient-level explainability
- [ ] Deploy as a web app using Streamlit or Flask
- [ ] Integrate with EHR (Electronic Health Records) data pipeline
- [ ] Add confidence intervals to model predictions
- [ ] Test for demographic bias across age groups

---

## 👤 Author

**[Your Name]**  
Internship Project — Machine Learning & Healthcare AI  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is created for educational and internship purposes.  
Dataset credit: [Diabetes Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
