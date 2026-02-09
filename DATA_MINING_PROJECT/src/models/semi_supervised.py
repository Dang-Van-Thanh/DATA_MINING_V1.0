import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import average_precision_score

class SemiSupervisedTrainer:
    def __init__(self, base_model, threshold=0.9):
        self.base_model = base_model
        self.threshold = threshold

    def run(self, X, y):
        # y: -1 = unlabeled
        clf = SelfTrainingClassifier(
            self.base_model,
            threshold=self.threshold
        )
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)[:, 1]
        return clf, y_proba
    
    def learning_curve(self, X, y, label_ratios):
        results = []

        for r in label_ratios:
            n = int(len(y) * r)
            y_semi = np.full_like(y, -1)
            idx = np.random.choice(len(y), n, replace=False)
            y_semi[idx] = y[idx]

            _, y_proba = self.run(X, y_semi)

            results.append({
                "label_ratio": r,
                "mean_probability": np.mean(y_proba)
            })

        return results
