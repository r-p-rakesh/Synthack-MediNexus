# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
from models.kidney_disease import train_model as train_kidney_model, predict_kidney_disease
from models.heart_disease import train_model as train_heart_model, predict_heart_disease
from models.breast_cancer import train_and_save_model as train_breast_model, predict_breast_cancer

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap
import lime
import lime.lime_tabular

app = Flask(__name__)

try:
    # Load breast cancer model and data
    bc_model, bc_X_test, bc_y_test, bc_shap_values, bc_accuracy, bc_feature_names, bc_explainer, bc_feature_explanations, bc_scaler = train_breast_model()
    breast_model = bc_model
    breast_scaler = bc_scaler
    breast_features = bc_feature_names
    print("Breast cancer model loaded successfully")
except Exception as e:
    print(f"Error loading breast cancer model: {str(e)}")
    bc_model = None
    bc_X_test = None
    bc_y_test = None
    bc_shap_values = None
    bc_accuracy = 0
    bc_feature_names = []
    bc_explainer = None
    bc_feature_explanations = {}
    bc_scaler = None
    breast_model = None
    breast_scaler = None
    breast_features = []

# Load kidney disease model and data
kd_model, kd_X_test, kd_y_test, kd_shap_values, kd_accuracy, kd_feature_names, kd_explainer, kd_feature_explanations, kd_scaler, kd_imputer = train_kidney_model()

# Load heart disease model and data
hd_model, hd_X_test, hd_y_test, hd_shap_values, hd_accuracy, hd_feature_names, hd_explainer, hd_feature_explanations, hd_scaler, hd_imputer, hd_cv_accuracy = train_heart_model()

# Load models and data
kidney_model = joblib.load('models/kidney_disease_model.pkl')
heart_model = joblib.load('models/heart_disease_model.pkl')
kidney_scaler = joblib.load('models/kidney_disease_scaler.pkl')
heart_scaler = joblib.load('models/heart_disease_scaler.pkl')
heart_imputer = joblib.load('models/heart_disease_imputer.pkl')

# Load feature names
with open('models/kidney_disease_features.json', 'r') as f:
    kidney_features = json.load(f)
with open('models/heart_disease_features.json', 'r') as f:
    heart_features = json.load(f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/home")
def home_page():
    return redirect(url_for('home'))

@app.route("/diagnose")
def diagnose():
    return render_template("index.html", 
                         total_samples_breast=len(bc_X_test) if bc_X_test is not None else 0,
                         total_samples_kidney=len(kd_X_test),
                         total_samples_heart=len(hd_X_test),
                         columns_breast=breast_features,
                         columns_kidney=kidney_features,
                         columns_heart=heart_features)

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")

@app.route("/signin")
def signin():
    return render_template("signin.html")

@app.route("/result")
def result():
    try:
        # Get the prediction data from the request
        prediction_data = request.args.get('data')
        if prediction_data:
            try:
                data = json.loads(prediction_data)
                # Handle both prediction types
                if 'error' in data:
                    return render_template("result.html", 
                                        result=data['error'],
                                        features=[],
                                        accuracy=0)
                
                # Format the prediction result
                prediction_text = data.get('prediction', '')
                if 'ckd' in prediction_text.lower():
                    prediction_text = 'Chronic Kidney Disease' if prediction_text == 'ckd' else 'No Kidney Disease'
                elif 'malignant' in prediction_text.lower():
                    prediction_text = 'Malignant (cancerous)' if 'malignant' in prediction_text.lower() else 'Benign (non-cancerous)'

                return render_template("result.html", 
                                    result=prediction_text,
                                    features=data.get('features', []),
                                    accuracy=data.get('accuracy', 0))
            except json.JSONDecodeError:
                return render_template("result.html", 
                                    result="Invalid prediction data",
                                    features=[],
                                    accuracy=0)
    except Exception as e:
        print(f"Error in result route: {str(e)}")
        return render_template("result.html", 
                            result="An error occurred while processing the prediction",
                            features=[],
                            accuracy=0)
    
    return render_template("result.html", 
                         result="No prediction data available",
                         features=[],
                         accuracy=0)

@app.route("/kidney")
def kidney_form():
    return render_template("kidney.html", columns=kidney_features)

@app.route("/heart")
def heart_form():
    return render_template("heart.html", columns=heart_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sample_index = int(request.form.get("sample"))
        if sample_index is None:
            return jsonify({
                "error": "Please provide a sample index",
                "prediction": None,
                "features": [],
                "accuracy": kd_accuracy
            })

        if sample_index < 0 or sample_index >= len(kd_X_test):
            return jsonify({
                "error": f"Sample index must be between 0 and {len(kd_X_test)-1}",
                "prediction": None,
                "features": [],
                "accuracy": kd_accuracy
            })

        sample = kd_X_test.iloc[[sample_index]]
        prediction = kd_model.predict(sample)[0]
        shap_sample = kd_shap_values[sample_index]

        result = "Kidney Disease Detected" if prediction == 1 else "No Kidney Disease"

        contributions = list(zip(kd_feature_names, shap_sample.values, sample.values[0]))
        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]

        reasons = []
        for feature, shap_val, feature_val in top_features:
            direction = "+" if shap_val > 0 else "-"
            reason = kd_feature_explanations.get(feature, "")
            reasons.append({
                "feature": feature,
                "value": round(feature_val, 2),
                "contribution": round(shap_val, 4),
                "direction": direction,
                "reason": reason
            })

        return jsonify({
            "prediction": result,
            "features": reasons,
            "accuracy": kd_accuracy
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            "error": "An error occurred during prediction",
            "prediction": None,
            "features": [],
            "accuracy": kd_accuracy
        })

@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Ensure all required features are present
        required_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400

        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Scale the features
        input_df = pd.DataFrame(kidney_scaler.transform(input_df), columns=kidney_features)
        
        # Make prediction
        prediction = kidney_model.predict(input_df)[0]
        probability = kidney_model.predict_proba(input_df)[0][1]
        
        # Get SHAP values for explanation
        shap_values = kd_explainer.shap_values(input_df)
        
        # Get feature contributions
        contributions = list(zip(kidney_features, shap_values[0], input_df.values[0]))
        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Format feature contributions
        feature_contributions = []
        for feature, shap_val, feature_val in top_features:
            feature_contributions.append({
                "feature": feature,
                "value": float(shap_val),
                "description": kd_feature_explanations.get(feature, "")
            })
        
        result_data = {
            "prediction": int(prediction),
            "confidence": float(probability),
            "feature_contributions": feature_contributions,
            "accuracy": kd_accuracy
        }
        
        return render_template('kidney_result.html', data=result_data)
    except Exception as e:
        print(f"Error in kidney prediction: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Ensure all required features are present
        required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400

        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Scale the features
        input_df = pd.DataFrame(heart_scaler.transform(input_df), columns=heart_features)
        
        # Make prediction
        prediction = heart_model.predict(input_df)[0]
        probability = heart_model.predict_proba(input_df)[0][1]
        
        # Get SHAP values for explanation
        shap_values = hd_explainer.shap_values(input_df)
        
        # Get feature contributions
        contributions = list(zip(heart_features, shap_values[0], input_df.values[0]))
        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Format feature contributions
        feature_contributions = []
        for feature, shap_val, feature_val in top_features:
            feature_contributions.append({
                "feature": feature,
                "value": float(shap_val),
                "description": hd_feature_explanations.get(feature, "")
            })
        
        result_data = {
            "prediction": int(prediction),
            "confidence": float(probability),
            "feature_contributions": feature_contributions,
            "accuracy": hd_accuracy
        }
        
        return render_template('heart_result.html', data=result_data)
    except Exception as e:
        print(f"Error in heart prediction: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/predict_breast", methods=["POST"])
def predict_breast():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Ensure all required features are present
        required_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean']
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400

        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Scale the features
        input_df = pd.DataFrame(breast_scaler.transform(input_df), columns=breast_features)
        
        # Make prediction
        prediction = breast_model.predict(input_df)[0]
        probability = breast_model.predict_proba(input_df)[0][1]
        
        # Get SHAP values for explanation
        shap_values = bc_explainer.shap_values(input_df)
        
        # Get feature contributions
        contributions = list(zip(breast_features, shap_values[0], input_df.values[0]))
        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Format feature contributions
        feature_contributions = []
        for feature, shap_val, feature_val in top_features:
            feature_contributions.append({
                "feature": feature,
                "value": float(shap_val),
                "description": bc_feature_explanations.get(feature, "")
            })
        
        return jsonify({
            "prediction": int(prediction),
            "confidence": float(probability),
            "feature_contributions": feature_contributions
        })
    except Exception as e:
        print(f"Error in breast cancer prediction: {str(e)}")
        return jsonify({
            "error": str(e)  # Return the actual error message
        }), 500

@app.route('/explanation')
def explanation():
    return render_template('explanation.html')

@app.route('/api/feature_importance/<model_type>')
def get_feature_importance(model_type):
    if model_type == 'kidney':
        model = kidney_model
        features = kidney_features
    elif model_type == 'heart':
        model = heart_model
        features = heart_features
    else:
        return jsonify({
            "error": "Invalid model type"
        })
    
    # Calculate feature importance
    importance = model.feature_importances_
    feature_importance = dict(zip(features, importance))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return jsonify(sorted_importance)

@app.route('/api/model_performance/<model_type>')
def get_model_performance(model_type):
    if model_type == 'kidney':
        accuracy = kd_accuracy
        cv_accuracy = None
    elif model_type == 'heart':
        accuracy = hd_accuracy
        cv_accuracy = hd_cv_accuracy
    else:
        return jsonify({
            "error": "Invalid model type"
        })
    
    return jsonify({
        "accuracy": accuracy,
        "cv_accuracy": cv_accuracy
    })

@app.route('/api/what_if', methods=['POST'])
def what_if_analysis():
    data = request.get_json()
    model_type = data.get('model_type')
    features = data.get('features')
    
    if model_type == 'kidney':
        model = kidney_model
        scaler = kidney_scaler
    elif model_type == 'heart':
        model = heart_model
        scaler = heart_scaler
        imputer = heart_imputer
    else:
        return jsonify({
            "error": "Invalid model type"
        })
    
    # Scale features
    input_df = pd.DataFrame([features])
    if model_type == 'heart':
        input_df = pd.DataFrame(imputer.transform(input_df), columns=features.keys())
    input_df = pd.DataFrame(scaler.transform(input_df), columns=features.keys())
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return jsonify({
        "prediction": prediction,
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)
