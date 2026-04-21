from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def get_models(random_state: int):
    """Return multiple models for benchmarking"""
    return {
        "logreg": LogisticRegression(max_iter=2000, random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "dt": DecisionTreeClassifier(random_state=random_state),
        "gb": GradientBoostingClassifier(random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state)
    }
    