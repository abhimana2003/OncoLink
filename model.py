from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


INPUT_DIR = "outputs_metabric"
RESULTS_DIR = os.path.join(INPUT_DIR, "model_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load processed METABRIC datasets.
    """
    X_all = pd.read_csv(os.path.join(INPUT_DIR, "X_all_genes.csv"))
    X_top = pd.read_csv(os.path.join(INPUT_DIR, "X_top_variable_genes.csv"))
    X_pca_20 = pd.read_csv(os.path.join(INPUT_DIR, "X_pca_20.csv"))
    X_clinical = pd.read_csv(os.path.join(INPUT_DIR, "X_clinical.csv"))
    y = pd.read_csv(os.path.join(INPUT_DIR, "y_labels.csv")).iloc[:, 0]

    return X_all, X_top, X_pca_20, X_clinical, y


def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name: str,
    feature_name: str,
) -> Dict[str, float]:
    """
    Fit model, evaluate performance, and save plots.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== {model_name} | {feature_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix: {model_name} | {feature_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"confusion_{model_name.lower().replace(' ', '_')}_{feature_name.lower().replace(' ', '_')}.png",
        )
    )
    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve: {model_name} | {feature_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"roc_{model_name.lower().replace(' ', '_')}_{feature_name.lower().replace(' ', '_')}.png",
        )
    )
    plt.close()

    return {
        "Model": model_name,
        "Features": feature_name,
        "Accuracy": acc,
        "F1": f1,
        "ROC_AUC": auc,
    }


def run_experiments() -> pd.DataFrame:
    """
    Run experiments on METABRIC feature sets.
    """
    X_all, X_top, X_pca_20, X_clinical, y = load_data()

    feature_sets: List[Tuple[str, pd.DataFrame]] = [
        ("Clinical Only", X_clinical),
        ("Top Features", X_top),
        ("PCA 20", X_pca_20),
        ("All Features", X_all),
    ]

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, C=0.1, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    results: List[Dict[str, float]] = []

    for feature_name, X in feature_sets:
        X_train, X_test, y_train, y_test = make_split(X, y)

        for model_name, model in models:
            result = evaluate_model(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=model_name,
                feature_name=feature_name,
            )
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)

    results_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    print("\nSaved results to:", results_path)
    print("\nFinal comparison table:")
    print(results_df)

    return results_df


if __name__ == "__main__":
    run_experiments()