ML Churn Prediction Pipeline

An end-to-end machine learning pipeline for customer churn prediction using scikit-learn, featuring:

- Multi-model benchmarking
- Stratified cross-validation
- Automatic best model selection
- Model + metrics + visualization export

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

or Specific dataset and target column:

```bash
python src/train.py --csv-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --target-column Churn
```

## Output

After training, files will be saved in `artifacts/`:

* model.joblib
* metrics.json
* test_sample.csv

and in `outputs/`:

* model_comparison.png


## Notes

* Target column will be renamed to `label`
* Supports numeric and categorical features

## Future Improvement

* Hyperparameter tuning
* Data validation
