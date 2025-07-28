# Employee Salary Prediction Dashboard

A Streamlit web application that lets you:

1. **Explore** the UCI Census Income dataset (`adult 3.csv`).
2. **Train** multiple classification algorithms (Logistic Regression, Random Forest, Gradient Boosting) on the cleaned data.
3. **Predict** whether a new employee earns `>50K` using the trained models via an interactive form.

---

## 📂 Project Structure

```
.
├── salary_dashboard/
│   ├── __init__.py
│   ├── app.py                # Streamlit entry-point
│   ├── data_loader.py        # Dataset loading helper
│   ├── preprocessing.py      # Cleaning & preprocessing utilities
│   ├── models.py             # Training, saving, loading models
│   ├── artifacts/            # Auto-generated trained model files
│   └── pages/
│       ├── 1_EDA_Dataset.py      # Dataset preview & EDA page
│       ├── 2_Model_Training.py   # Model training page
│       └── 3_Predict_Salary.py   # Prediction UI page
├── adult 3.csv              # Raw dataset (ensure this stays in project root)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Launch the dashboard**

```bash
streamlit run salary_dashboard/app.py
```

The browser will open automatically (or visit the displayed URL) with three sidebar pages: Dataset preview, Model training, and Predict Salary.

---

## 🔧 Usage Tips

* Train the models once on the *Model Training* page; trained pipelines are saved to `salary_dashboard/artifacts/`.
* After training, the *Predict Salary* page loads the saved models for instant predictions.
* You can retrain at any time using different `test_size` or `random_state` values; artifacts are overwritten.

---

## 📊 Algorithms Implemented

| Algorithm | Library | Notes |
|-----------|---------|-------|
| Logistic Regression | scikit-learn | baseline linear classifier |
| Random Forest | scikit-learn | ensemble of decision trees |
| Gradient Boosting | scikit-learn | additive ensemble, handles non-linearities |
| Support Vector Machine | scikit-learn | RBF kernel with probability estimates |
| K-Nearest Neighbors | scikit-learn | K=10, non-parametric |
| XGBoost | xgboost | gradient-boosted decision trees (requires `xgboost` wheel) |

During training the dashboard reports Accuracy, Precision, Recall, F1 and shows Confusion-Matrix heatmaps for each model.

---

## 🖼️ Visualisations

* Interactive Altair histograms for numeric columns.
* Category count bars for categorical columns.
* Metric bar-chart comparing model scores.
* Seaborn heatmap confusion matrices.

---

## 🛠️ Tech Stack

* Python ≥ 3.9
* Streamlit
* Pandas, NumPy
* scikit-learn

---

## 🤝 Contributing

Pull requests are welcome! Open an issue first to discuss any major changes.

---

## 📄 License

This project is provided for educational purposes and comes with no warranty.
