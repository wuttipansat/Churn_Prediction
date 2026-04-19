from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_model(model_name: str, random_state: int):
    """Select Basic model including Logistic Regression and Random Forest Classification."""
    if model_name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    return ValueError(f"Unsupported model: {model_name}")
