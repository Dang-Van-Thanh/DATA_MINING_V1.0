import yaml
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import AssociationMiner
from src.mining.clustering import ClusteringModel
from src.models.supervised import SupervisedTrainer
from src.models.semi_supervised import SemiSupervisedTrainer


# ===============================
# UTILS
# ===============================
def load_params(path="configs/params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths: dict):
    for p in paths.values():
        os.makedirs(p, exist_ok=True)


# ===============================
# MAIN PIPELINE
# ===============================
def main():
    print("🚀 Loading parameters...")
    params = load_params()
    np.random.seed(params["seed"])

    # ---------------------------
    # PATHS
    # ---------------------------
    raw_path = params["paths"]["raw_data"]
    processed_path = params["paths"]["processed_data"]
    output_paths = params["paths"]["outputs"]
    ensure_dirs(output_paths)

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    print("📥 Loading raw data...")
    df = DataLoader(raw_path).load()

    # ---------------------------
    # CLEAN DATA
    # ---------------------------
    print("🧹 Cleaning data...")
    cleaner = DataCleaner(
        drop_columns=params["data"]["drop_columns"]
    )
    df = cleaner.clean(df)

    # ---------------------------
    # TARGET ENCODING
    # ---------------------------
    target = params["data"]["target"]
    df[target] = df[target].map({"yes": 1, "no": 0})

    df.to_csv(processed_path, index=False)

    # ===============================
    # ASSOCIATION RULE MINING
    # ===============================
    if params["association"]["enabled"]:
        print("🔗 Association rule mining...")
        assoc_cols = params["data"]["categorical_cols"] + [target]
        df_assoc = df[assoc_cols].dropna()

        encoder = OneHotEncoder(sparse=False)
        assoc_matrix = encoder.fit_transform(df_assoc)
        assoc_df = pd.DataFrame(
            assoc_matrix,
            columns=encoder.get_feature_names_out(assoc_cols)
        )

        miner = AssociationMiner(
            min_support=params["association"]["min_support"],
            min_confidence=params["association"]["min_confidence"],
            min_lift=params["association"]["min_lift"]
        )

        rules = miner.run(assoc_df)
        rules.to_csv(
            os.path.join(output_paths["tables"], "association_rules.csv"),
            index=False
        )

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    print("🧱 Building feature pipeline...")
    X = df.drop(columns=[target])
    y = df[target]

    fb = FeatureBuilder(
        num_cols=params["data"]["numerical_cols"],
        cat_cols=params["data"]["categorical_cols"],
        scaling=params["preprocessing"]["scaling"]
    )

    pipeline = fb.build_pipeline()
    X_transformed = pipeline.fit_transform(X)

    # Save pipeline for Streamlit
    joblib.dump(
        pipeline,
        os.path.join(output_paths["models"], "feature_pipeline.pkl")
    )

    # ===============================
    # CLUSTERING
    # ===============================
    if params["clustering"]["enabled"]:
        print("👥 Clustering customers...")
        clusterer = ClusteringModel(
            k_range=params["clustering"]["k_range"]
        )
        best_k, scores = clusterer.find_best_k(X_transformed)
        model_cluster, labels = clusterer.fit(X_transformed, best_k)

        df["cluster"] = labels
        df.to_csv(
            os.path.join(output_paths["tables"], "clustered_data.csv"),
            index=False
        )

        print(f"   → Best K = {best_k}")

    # ===============================
    # SUPERVISED LEARNING
    # ===============================
    print("🤖 Training supervised models...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=params["preprocessing"]["test_size"],
        stratify=y if params["preprocessing"]["stratify"] else None,
        random_state=params["seed"]
    )

    trainer = SupervisedTrainer(params["modeling"]["models"])
    results = trainer.train_eval(X_train, X_test, y_train, y_test)

    # Select best model
    rows = []
    best_model = None
    best_pr_auc = -1
    best_model_name = None

    for name, res in results.items():
        rows.append({
            "model": name,
            "f1": res["f1"],
            "pr_auc": res["pr_auc"]
        })

        if res["pr_auc"] > best_pr_auc:
            best_pr_auc = res["pr_auc"]
            best_model = res["model"]
            best_model_name = name

    results_df = pd.DataFrame(rows)
    results_df.to_csv(
        os.path.join(output_paths["tables"], "supervised_results.csv"),
        index=False
    )

    # Save best model for Streamlit
    joblib.dump(
        best_model,
        os.path.join(output_paths["models"], "best_model.pkl")
    )

    with open(os.path.join(output_paths["models"], "model_info.txt"), "w") as f:
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"PR-AUC: {best_pr_auc:.4f}\n")

    print(f"🏆 Best model: {best_model_name} | PR-AUC = {best_pr_auc:.4f}")

    # ===============================
    # SEMI-SUPERVISED LEARNING
    # ===============================
    if params["semi_supervised"]["enabled"]:
        print("🧪 Semi-supervised learning...")
        semi_rows = []

        for ratio in params["semi_supervised"]["label_ratios"]:
            n_labeled = int(len(y) * ratio)

            y_semi = np.full_like(y, -1)
            labeled_idx = np.random.choice(len(y), n_labeled, replace=False)
            y_semi[labeled_idx] = y.iloc[labeled_idx]

            semi = SemiSupervisedTrainer(
                base_model=best_model,
                threshold=params["semi_supervised"]["confidence_threshold"]
            )

            clf, y_proba = semi.run(X_transformed, y_semi)

            semi_rows.append({
                "label_ratio": ratio,
                "mean_pseudo_probability": np.mean(y_proba)
            })

        pd.DataFrame(semi_rows).to_csv(
            os.path.join(output_paths["tables"], "semi_supervised_results.csv"),
            index=False
        )

    print("✅ FULL PIPELINE FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
