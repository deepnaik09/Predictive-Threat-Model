import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
file_path = "dataSet/newPreprocessed.csv"
mergedData = pd.read_csv(file_path)

# Encode categorical variable
label_encoder = LabelEncoder()
mergedData["transaction_type"] = label_encoder.fit_transform(mergedData["transaction_type"])

# Handle missing values
mergedData = mergedData.drop_duplicates()
mergedData.columns = mergedData.columns.str.strip()
mergedData = mergedData.fillna(mergedData.mean(numeric_only=True))

# Select features and target
X = mergedData[['rental_agreement', 'transaction_type', 'avg_monthly_balance', 'gst_filing_status',
                'overdraft_frequency', 'salary_deposits', 'guarantor_exists',
                'upi_transaction_count', 'age']]
y = mergedData["loan_approved"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save models
joblib.dump(dt_model, "decision_tree.pkl")
joblib.dump(rf_model, "random_forest.pkl")
joblib.dump(log_reg, "logistic_regression.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_choice = data.get("model", "random_forest")
    
    input_features = np.array(data["features"]).reshape(1, -1)
    scaler = joblib.load("scaler.pkl")
    input_features = scaler.transform(input_features)
    
    if model_choice == "decision_tree":
        model = joblib.load("decision_tree.pkl")
    elif model_choice == "logistic_regression":
        model = joblib.load("logistic_regression.pkl")
    elif model_choice == "svm":
        model = joblib.load("svm_model.pkl")
    else:
        model = joblib.load("random_forest.pkl")
    
    prediction = model.predict(input_features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
