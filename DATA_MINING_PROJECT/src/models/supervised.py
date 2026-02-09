from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score

# Optional dependency: xgboost (not required to run the pipeline)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

class SupervisedTrainer:
    def __init__(self, params):
        self.params = params

    def get_models(self):
        models = {}

        if self.params["logistic_regression"]["enabled"]:
            models["logreg"] = LogisticRegression(
                max_iter=self.params["logistic_regression"]["max_iter"],
                class_weight=self.params["logistic_regression"]["class_weight"],
                solver="lbfgs",
                random_state=42
            )

        if self.params["random_forest"]["enabled"]:
            models["rf"] = RandomForestClassifier(
                n_estimators=self.params["random_forest"]["n_estimators"],
                class_weight=self.params["random_forest"]["class_weight"],
                random_state=42,
                n_jobs=-1
            )

        if HAS_XGB and self.params["xgboost"]["enabled"]:
            models["xgb"] = XGBClassifier(
                n_estimators=self.params["xgboost"]["n_estimators"],
                max_depth=self.params["xgboost"]["max_depth"],
                learning_rate=self.params["xgboost"]["learning_rate"],
                eval_metric="logloss",
                random_state=42
            )

        return models

    def train_eval(self, X_train, X_test, y_train, y_test):
        results = {}
        models = self.get_models()

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            results[name] = {
                "f1": f1_score(y_test, y_pred),
                "pr_auc": average_precision_score(y_test, y_proba),
                "model": model
            }

        return results
