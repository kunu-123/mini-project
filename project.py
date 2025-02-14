import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model and Evaluation Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Explainability Imports
import shap
from lime.lime_tabular import LimeTabularExplainer

# Try importing SMOTE from imblearn, otherwise we will use class_weight
try:
    from imblearn.over_sampling import SMOTE
    smote_available = True
except ImportError:
    print("imbalanced-learn not installed. Falling back to class_weight based handling.")
    smote_available = False


# ===== Utility Functions =====

def preprocess_dataset(df, target_column, id_columns=[], categorical_cols=None):
    """
    Preprocess the dataset by:
      - Dropping ID columns,
      - Filling missing values (median for numeric, mode for categorical),
      - Encoding categorical columns,
      - Scaling numeric features.
    """
    # Drop ID columns if present
    if id_columns:
        df = df.drop(columns=[col for col in id_columns if col in df.columns])
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            # Avoid chained assignment: assign result back to the DataFrame column
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical columns using LabelEncoder
    # If categorical_cols is not given, assume all object dtype columns are categorical.
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, encoder, scaler


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train three classifiers and evaluate:
       - Logistic Regression,
       - Random Forest,
       - XGBoost.
    Returns a dictionary of trained models and predictions.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluation metrics
        print(f"\n{name} - Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"{name} - Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"{name} - ROC AUC Score: {roc_auc:.2f}")
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.show()
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "roc_auc": roc_auc
        }
    return results


def explain_model_shap(model, X, dataset_name, model_name):
    """
    Generate a SHAP summary plot for model explainability.
    """
    print(f"Generating SHAP summary plot for {model_name} on {dataset_name}...")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary Plot - {model_name} on {dataset_name}")
    plt.show()


def explain_model_lime(model, X_train, X_instance, feature_names, dataset_name, model_name):
    """
    Generate a LIME explanation for a single instance.
    """
    print(f"Generating LIME explanation for {model_name} on {dataset_name}...")
    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        mode="classification"
    )
    exp = lime_explainer.explain_instance(X_instance, model.predict_proba, num_features=10)
    exp.show_in_notebook(show_table=True)
    # If not in notebook, you can save the explanation as HTML:
    # exp.save_to_file(f"lime_explanation_{dataset_name}_{model_name}.html")


# ===== Main Pipeline =====

# Define information for each dataset:
# For each dataset, provide:
#  - file_path: location of CSV file
#  - target_column: name of the target variable
#  - id_columns: any columns to drop (IDs)
#  - categorical_cols: list of categorical columns (if needed, else None to auto-detect)

datasets_info = {
    "Telco Customer Churn": {
        "file_path": "Telco-Customer-Churn.csv",  # Update with your local path if needed
        "target_column": "Churn",
        "id_columns": ["customerID"],
        "categorical_cols": None  # Auto-detect if not provided
    },
    "Credit Card Fraud": {
        "file_path": "creditcard.csv",  # Update with your local path if needed
        "target_column": "Class",
        "id_columns": [],
        "categorical_cols": None
    },
    "Employee Attrition": {
        "file_path": "HR-Employee-Attrition.csv",  # Update with your local path if needed
        "target_column": "Attrition",  # Adjust if the target variable name differs
        "id_columns": ["EmployeeNumber"],
        "categorical_cols": None
    },
    "Loan Approval": {
        "file_path": "loan_data.csv",  # Using the training portion of Loan Approval
        "target_column": "Loan_Status",
        "id_columns": ["Loan_ID"],
        "categorical_cols": None
    }
}

# Loop through each dataset and run the pipeline
for dataset_name, info in datasets_info.items():
    print(f"\n\n=== Processing: {dataset_name} ===")
    
    # Load dataset
    df = pd.read_csv(info["file_path"])
    
    # Preprocess the dataset
    X, y, le, scaler = preprocess_dataset(
        df, 
        target_column=info["target_column"], 
        id_columns=info.get("id_columns", []), 
        categorical_cols=info.get("categorical_cols", None)
    )
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # If SMOTE is available, apply it to the training data.
    if smote_available:
        print("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    else:
        print("Using class_weight in models to handle imbalance.")
    
    # Train and evaluate models
    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # For explainability, use the first model (e.g., Random Forest) as an example.
    # Also, for LIME, we need the original feature names:
    feature_names = pd.read_csv(info["file_path"]).drop(columns=info.get("id_columns", [])).drop(columns=[info["target_column"]]).columns.tolist()
    
    # Choose one model to explain; here we choose Random Forest if available.
    if "Random Forest" in model_results:
        rf_model = model_results["Random Forest"]["model"]
        # SHAP explanation
        explain_model_shap(rf_model, X_test, dataset_name, "Random Forest")
        # LIME explanation on the first instance of test data
        explain_model_lime(rf_model, X_train, X_test[0], feature_names, dataset_name, "Random Forest")
    
    print(f"=== Finished processing {dataset_name} ===\n")