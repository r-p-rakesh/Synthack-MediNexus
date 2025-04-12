# models/kidney_disease.py
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
import os

def train_and_save_model():
    # Load and preprocess the dataset
    df = pd.read_csv("datasets/kidney_disease.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df[df['classification'].notna()]
    df['classification'] = df['classification'].astype(str).str.strip().replace({'ckd': 1, 'notckd': 0, 'ckd\t': 1})
    y = df['classification'].astype(int)
    df = df.drop(['id'], axis=1, errors='ignore')
    X = df.drop('classification', axis=1)

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype(str).str.strip()
        X[col] = LabelEncoder().fit_transform(X[col])

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.5,
        reg_lambda=1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save model and scaler
    joblib.dump(model, 'models/kidney_disease_model.pkl')
    joblib.dump(scaler, 'models/kidney_disease_scaler.pkl')
    joblib.dump(imputer, 'models/kidney_disease_imputer.pkl')

    # Save feature names
    with open('models/kidney_disease_features.json', 'w') as f:
        json.dump(X.columns.tolist(), f)

    explainer = shap.Explainer(model, X_train)

    feature_explanations = {
        # Add feature explanations if needed
    }

    return (
        model,
        X_test,
        y_test,
        explainer(X_test),
        accuracy_score(y_test, model.predict(X_test)),
        X.columns.tolist(),
        explainer,
        feature_explanations,
        scaler,
        imputer
    )

# Load or train model
try:
    model = joblib.load('models/kidney_disease_model.pkl')
    scaler = joblib.load('models/kidney_disease_scaler.pkl')
    imputer = joblib.load('models/kidney_disease_imputer.pkl')
    with open('models/kidney_disease_features.json', 'r') as f:
        feature_names = json.load(f)
    explainer = shap.Explainer(model)
except (FileNotFoundError, Exception):
    model, X_test, y_test, shap_values, accuracy, feature_names, explainer, feature_explanations, scaler, imputer = train_and_save_model()

def train_model():
    try:
        return (
            model,
            X_test,
            y_test,
            shap_values,
            accuracy,
            feature_names,
            explainer,
            feature_explanations,
            scaler,
            imputer
        )
    except NameError:
        return train_and_save_model()

def predict_kidney_disease(user_input, model, explainer, scaler, imputer, feature_names):
    input_df = pd.DataFrame([user_input])

    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = input_df[col].astype(str).str.strip()
        input_df[col] = LabelEncoder().fit(input_df[col]).transform(input_df[col])

    input_df = pd.DataFrame(imputer.transform(input_df), columns=feature_names)
    input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

    prediction = model.predict(input_df)[0]
    shap_values = explainer(input_df)
    shap_sample = shap_values[0]

    result = "Kidney Disease Detected" if prediction == 1 else "No Kidney Disease"

    contributions = list(zip(feature_names, shap_sample.values, input_df.values[0]))
    top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]

    return result, top_features

if __name__ == "__main__":
    train_and_save_model()
