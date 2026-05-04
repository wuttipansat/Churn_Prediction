# Production-ready Machine Learning API for customer churn prediction

An end-to-end Machine Learning project predicts whether a customer is likely to churn using the Telcom Customer Churn dataset.
The project includes data preprocessing, feature engineering, multi-model training and comparison, FastAPI interface with Docker deployment.

## Features

* Customer churn prediction with Machine Learning models
* Automated preprocessing and feature engineering
* Multi-model training and evaluation pipeline
* Saved model artifact using joblib
* REST API powered by FastAPI
* Dockerized deployment workflow


## Models
* Logistic Regression
* Random Forest Classifier
* Decision Tree Classifier
* Gradient Boosting Classifier
* Support Vector Machine
* etc. (More model will be added)

## Getting Started

### 1.Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python src/train.py
```

or Specific dataset and target column:

```bash
python src/train.py --csv-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --target-column Churn
```

After training, files will be saved in `artifacts/`:

## Run the API Locally

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

Open the API documentation:

```
http://localhost:8000/docs
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t churn-prediction-api .
```

### Run API Container

```bash
docker run --rm -p 8000:8000 churnguard-api
```

Open API documentation:

```
http://localhost:8000/docs
```

## Test the API

Start the API first, then run:

```bash
python app/test_api.py
```

## Author

Developed by **Wuttipan Satienpaisan**