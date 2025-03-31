
---

```markdown
# 📊 Customer Churn Prediction Using Machine Learning

This project aims to predict **customer churn** for a telecommunications company using machine learning techniques. By applying preprocessing, feature engineering, and model tuning, we evaluate two models and determine that **XGBoost** performs the best with outstanding accuracy.

---

## 📁 Project Overview

- Exploratory Data Analysis (EDA) and Visualization
- Feature Encoding (OneHotEncoder)
- Model Training (Logistic Regression & XGBoost)
- Model Comparison (ROC AUC Score)
- Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV)
- Feature Importance Visualization (XGBoost)

---

## 📦 Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

---

## 🧹 Data Preprocessing

- Data was loaded from `customer_churn_dataset-testing-master.csv`.
- Missing values were checked and handled.
- Categorical variables were encoded using `OneHotEncoder`.
- Irrelevant columns such as `CustomerID`, `Age`, and the target `Churn` were excluded from the features.
- The target variable (`y`) is the `Churn` column.

---

## 🧠 Models Used

### 1. Logistic Regression
- Implemented with a `Pipeline` that includes `StandardScaler` and `LogisticRegression`.
- Tuned with `GridSearchCV` to find optimal parameters.

### 2. XGBoost Classifier
- Initially trained with default parameters.
- Further tuning performed using `RandomizedSearchCV`, but performance did not significantly improve — default model retained.

---

## 🧪 Model Performance

| Model               | Accuracy | ROC AUC Score |
|---------------------|----------|----------------|
| Logistic Regression | ~0.88    | **0.9067**     |
| XGBoost             | ~0.96    | **0.9978**     |

---

## 📈 ROC Curve

The ROC Curve provides a visual representation of classification performance.  
Below is a comparison between **Logistic Regression** and **XGBoost**:

![roc_curve](roc_curve.png)

> This image is generated during model evaluation and saved as `roc_curve.png`.

---

## 💡 Feature Importance (XGBoost)

Using XGBoost, we can assess which features contribute most to predicting churn:

```python
from xgboost import plot_importance
plot_importance(model)
```

This provides interpretability and helps in understanding model decisions.

---

## 🔧 Model Tuning

### Grid Search for Logistic Regression:

```python
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__max_iter': [1000, 2000]
}
```

### Randomized Search for XGBoost:

```python
params = {
    'gamma': uniform(0,0.5),
    'learning_rate': uniform(0.03,0.3),
    'max_depth': randint(2,6),
    'n_estimators': randint(100,150),
    'subsample': uniform(0.6,0.4)
}
```

---

## 📁 Project Structure

```
├── customer_churn_dataset-testing-master.csv
├── churn_prediction.ipynb
├── roc_curve.png
├── README.md
└── requirements.txt
```

---

## ✅ Conclusion

- Logistic Regression is a simple, interpretable model achieving around 90% ROC AUC.
- XGBoost significantly outperforms with ~96% accuracy and **99.77% ROC AUC**, making it the top-performing model.
- Since the dataset was balanced, no resampling techniques (e.g., SMOTE) were needed.
- Cross-validation confirms that the XGBoost model is **not overfitting** and generalizes well.


```
