# Basic ML Pipeline

A simple end-to-end machine learning pipeline using Python and scikit-learn.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

or use this as the example

```bash
python -m src.train --csv-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --target-column Churn
```

## Output

After training, files will be saved in `artifacts/`:

* model.joblib
* metrics.json

## Notes

* Target column will be renamed to `label`
* Supports numeric and categorical features
* Models: `logreg`, `rf`

## Future Improvement

* Add more models
* Hyperparameter tuning
* Model comparison
* Data validation
