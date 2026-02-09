import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ClusteringModel:
    def __init__(self, k_range):
        self.k_range = k_range

    def find_best_k(self, X):
        scores = {}
        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            scores[k] = score
        best_k = max(scores, key=scores.get)
        return best_k, scores

    def fit(self, X, k, return_score=False):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)

        if return_score:
            score = silhouette_score(X, labels)
            return model, labels, score

        return model, labels
