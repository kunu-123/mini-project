import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import shap

# Load datasets
train_path = "C:\\Users\\KIIT\\Desktop\\train_u6lujuX_CVtuZ9i.csv"
test_path = "C:\\Users\\KIIT\\Desktop\\test_Y3wMUE5_7gLdaTN.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Merge datasets (align test columns)
test_data["Loan_Status"] = np.nan  # Placeholder for test set
full_data = pd.concat([train_data, test_data], ignore_index=True)

# Data Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
full_data.fillna(method='ffill', inplace=True)
categorical_features = full_data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_features:
    full_data[col] = encoder.fit_transform(full_data[col].astype(str))

scaler = StandardScaler()
full_data.iloc[:, :-1] = scaler.fit_transform(full_data.iloc[:, :-1])

# Split back into train and test sets
train_data = full_data[full_data['Loan_Status'].notna()]
test_data = full_data[full_data['Loan_Status'].isna()].drop(columns=['Loan_Status'])

# Splitting the dataset
X = train_data.drop(columns=['Loan_Status'])
y = train_data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # ROC and Precision-Recall Curves
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f}')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'{name}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

# Explainable AI using SHAP
explainer = shap.Explainer(models["XGBoost"], X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
