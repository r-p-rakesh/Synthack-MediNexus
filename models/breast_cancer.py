# This file is kept as a placeholder but functionality is removed
# since breast cancer model is no longer needed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap
import json
import requests
from io import StringIO

def train_and_save_model():
    try:
        # Load the dataset from URL
        url = "https://raw.githubusercontent.com/dataprofessor/data/master/wisconsin.csv"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch dataset. Status code: {response.status_code}")
        
        # Read the CSV data
        df = pd.read_csv(StringIO(response.text))
        
        # Prepare features and target
        X = df.drop('diagnosis', axis=1)  # Assuming 'diagnosis' is the target column
        y = df['diagnosis'].map({'M': 1, 'B': 0})  # Convert M/B to 1/0
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test_scaled, y_test)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Create feature explanations
        feature_explanations = {
            'radius_mean': 'Mean radius of the tumor cells',
            'texture_mean': 'Mean texture of the tumor cells',
            'perimeter_mean': 'Mean perimeter of the tumor cells',
            'area_mean': 'Mean area of the tumor cells',
            'smoothness_mean': 'Mean smoothness of the tumor cells',
            'compactness_mean': 'Mean compactness of the tumor cells',
            'concavity_mean': 'Mean concavity of the tumor cells',
            'concave_points_mean': 'Mean number of concave portions of the contour',
            'symmetry_mean': 'Mean symmetry of the tumor cells',
            'fractal_dimension_mean': 'Mean fractal dimension of the tumor cells'
        }
        
        # Save the model and scaler
        joblib.dump(model, 'models/breast_cancer_model.pkl')
        joblib.dump(scaler, 'models/breast_cancer_scaler.pkl')
        
        # Save feature names
        with open('models/breast_cancer_features.json', 'w') as f:
            json.dump(feature_names, f)
        
        return model, X_test, y_test, shap_values, accuracy, feature_names, explainer, feature_explanations, scaler
    except Exception as e:
        print(f"Error in breast cancer model training: {str(e)}")
        raise

def predict_breast_cancer(input_data, model, scaler, explainer, feature_names, feature_explanations):
    try:
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        required_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean']
        for feature in required_features:
            if feature not in input_df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Scale the features
        input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # Get feature contributions
        contributions = list(zip(feature_names, shap_values[0], input_df.values[0]))
        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Format feature contributions
        feature_contributions = []
        for feature, shap_val, feature_val in top_features:
            feature_contributions.append({
                "feature": feature,
                "value": float(shap_val),
                "description": feature_explanations.get(feature, "")
            })
        
        return {
            "prediction": int(prediction),
            "confidence": float(probability),
            "feature_contributions": feature_contributions
        }
    except Exception as e:
        raise Exception(f"Error in breast cancer prediction: {str(e)}")

def predict(X, model, scaler):
    """Placeholder function - model not in use"""
    return None 