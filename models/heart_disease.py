# models/heart_disease.py
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

def train_and_save_model():
    # Load dataset
    df = pd.read_csv("datasets/heart_disease.csv")
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Feature scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost model
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
    joblib.dump(model, 'models/heart_disease_model.pkl')
    joblib.dump(scaler, 'models/heart_disease_scaler.pkl')
    joblib.dump(imputer, 'models/heart_disease_imputer.pkl')

    # Save feature names
    with open('models/heart_disease_features.json', 'w') as f:
        json.dump(X.columns.tolist(), f)

    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Cross-validation accuracy
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_accuracy = cv_scores.mean()

    # Feature explanations
    feature_explanations = {
        'age': 'Age of the patient in years',
        'sex': 'Gender of the patient (1 = male, 0 = female)',
        'cp': 'Chest pain type (1-4)',
        'trestbps': 'Resting blood pressure in mm Hg',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (1-3)',
        'ca': 'Number of major vessels colored by flourosopy (0-3)',
        'thal': 'Thalassemia type (3 = normal, 6 = fixed defect, 7 = reversable defect)'
    }

    return (
        model,
        X_test,
        y_test,
        shap_values,
        accuracy,
        X.columns.tolist(),
        explainer,
        feature_explanations,
        scaler,
        imputer,
        cv_accuracy
    )

# Load or train model
try:
    model = joblib.load('models/heart_disease_model.pkl')
    scaler = joblib.load('models/heart_disease_scaler.pkl')
    imputer = joblib.load('models/heart_disease_imputer.pkl')
    with open('models/heart_disease_features.json', 'r') as f:
        feature_names = json.load(f)
    explainer = shap.Explainer(model)
except (FileNotFoundError, Exception):
    model, X_test, y_test, shap_values, accuracy, feature_names, explainer, feature_explanations, scaler, imputer, cv_accuracy = train_and_save_model()

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
            imputer,
            cv_accuracy
        )
    except NameError:
        return train_and_save_model()

def predict_heart_disease(user_input, model, explainer, scaler, feature_names):
    input_df = pd.DataFrame([user_input])

    # Impute missing values
    input_df = pd.DataFrame(imputer.transform(input_df), columns=feature_names)

    # Scale features
    input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

    # Make prediction
    prediction = model.predict(input_df)[0]
    shap_values = explainer(input_df)
    shap_sample = shap_values[0]

    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    # Get top contributing features
    contributions = list(zip(feature_names, shap_sample.values, input_df.values[0]))
    top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]

    return result, top_features

if __name__ == "__main__":
    train_and_save_model() 