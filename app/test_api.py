import requests
import json

BASE_URL = "http://localhost:8000"

predict_url = f"{BASE_URL}/predict"

payload = {
        "records": [
            {
                "customerID": "TEST-001",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "One Year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 105.8,
                "TotalCharges": 1058.4
            }
        ]
    }


response = requests.post(predict_url, json=payload)

print("=== Predict ===")
print("Status Code:", response.status_code)
print(json.dumps(response.json(), indent=2))