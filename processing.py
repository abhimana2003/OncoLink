from __future__ import annotations

import os
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/METABRIC_RNA_Mutation.csv"
OUTPUT_DIR = "outputs_metabric"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("+", "plus", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def map_binary_column(series: pd.Series, positive_tokens: List[str], negative_tokens: List[str]) -> pd.Series:
    def _map_value(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in positive_tokens:
            return 1
        if s in negative_tokens:
            return 0
        return np.nan

    return series.map(_map_value)


def main() -> None:
    print("Loading METABRIC dataset...")
    df = pd.read_csv(DATA_PATH)
    df = clean_column_names(df)

    print("Raw shape:", df.shape)

    # ----------------------------
    # Define target
    # ----------------------------
    # Best simple binary target for this dataset
    target_col = "overall_survival"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    # ----------------------------
    # Define clinical columns
    # ----------------------------
    clinical_candidates = [
        "age_at_diagnosis",
        "chemotherapy",
        "hormone_therapy",
        "radio_therapy",
        "tumor_size",
        "tumor_stage",
        "lymph_nodes_examined_positive",
        "er_status",
        "her2_status",
        "pr_status",
        "neoplasm_histologic_grade",
    ]

    clinical_cols = [c for c in clinical_candidates if c in df.columns]

    # Map likely binary/string clinical columns
    for col in ["chemotherapy", "hormone_therapy", "radio_therapy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["er_status", "her2_status", "pr_status"]:
        if col in df.columns:
            df[col] = map_binary_column(
                df[col],
                positive_tokens=["positive", "pos", "1"],
                negative_tokens=["negative", "neg", "0"],
            )

    # numeric clinical columns
    for col in clinical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----------------------------
    # Define gene-expression columns
    # ----------------------------
    # Exclude metadata, target, and mutation columns
    metadata_exclude = {
        "patient_id",
        "type_of_breast_surgery",
        "cancer_type",
        "cancer_type_detailed",
        "cellularity",
        "pam50_plus_claudin_low_subtype",
        "pam50__claudin_low_subtype",
        "cohort",
        "er_status_measured_by_ihc",
        "her2_status_measured_by_snp6",
        "tumor_other_histologic_subtype",
        "inferred_menopausal_state",
        "integrative_cluster",
        "primary_tumor_laterality",
        "oncotree_code",
        "death_from_cancer",
        "3_gene_classifier_subtype",
        "3_gene_classifier_subtype".replace("3", "3"),  # harmless
        "overall_survival_months",
        target_col,
    }

    mutation_cols = [c for c in df.columns if c.endswith("_mut")]

    gene_cols = [
        c for c in df.columns
        if c not in metadata_exclude
        and c not in clinical_cols
        and c not in mutation_cols
    ]

    # force numeric for genes
    for col in gene_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----------------------------
    # Handle missing values
    # ----------------------------
    # Keep columns with at least 80% non-missing
    keep_threshold = int(0.8 * len(df))
    keep_cols = [c for c in clinical_cols + gene_cols + [target_col] if df[c].notna().sum() >= keep_threshold]
    df = df[keep_cols]

    clinical_cols = [c for c in clinical_cols if c in df.columns]
    gene_cols = [c for c in gene_cols if c in df.columns]

    # Fill missing clinical/gene values with column means
    for col in clinical_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in gene_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Drop duplicate rows if any
    df = df.drop_duplicates()

    # ----------------------------
    # Build feature matrices
    # ----------------------------
    X_genes = df[gene_cols]
    X_clinical = df[clinical_cols]
    X_all = pd.concat([X_genes, X_clinical], axis=1)
    y = df[target_col]

    print("\nFinal shapes:")
    print("Genes:", X_genes.shape)
    print("Clinical:", X_clinical.shape)
    print("Combined:", X_all.shape)
    print("Labels:", y.shape)

    print("\nClass balance:")
    print(y.value_counts())

    # ----------------------------
    # Scale
    # ----------------------------
    scaler_genes = StandardScaler()
    X_genes_scaled = scaler_genes.fit_transform(X_genes)

    scaler_clinical = StandardScaler()
    X_clinical_scaled = scaler_clinical.fit_transform(X_clinical)

    X_all_scaled = np.concatenate([X_genes_scaled, X_clinical_scaled], axis=1)

    # ----------------------------
    # Top variable genes
    # ----------------------------
    num_top_genes = min(1000, X_genes.shape[1])
    variances = np.var(X_genes_scaled, axis=0)
    top_indices = np.argsort(variances)[-num_top_genes:]
    X_top_genes = X_genes_scaled[:, top_indices]

    # Keep combined top genes + clinical
    X_top = np.concatenate([X_top_genes, X_clinical_scaled], axis=1)

    # Also save clinical-only
    X_clinical_only = X_clinical_scaled

    # ----------------------------
    # PCA on gene matrix
    # ----------------------------
    pca_20 = PCA(n_components=20, random_state=42)
    pca_50 = PCA(n_components=50, random_state=42)
    pca_var = PCA(n_components=0.95, random_state=42)

    X_pca_20_genes = pca_20.fit_transform(X_genes_scaled)
    X_pca_50_genes = pca_50.fit_transform(X_genes_scaled)
    X_pca_var_genes = pca_var.fit_transform(X_genes_scaled)

    # combined PCA-gene + clinical
    X_pca_20 = np.concatenate([X_pca_20_genes, X_clinical_scaled], axis=1)
    X_pca_50 = np.concatenate([X_pca_50_genes, X_clinical_scaled], axis=1)
    X_pca_var = np.concatenate([X_pca_var_genes, X_clinical_scaled], axis=1)

    print("\nExplained variance:")
    print(f"PCA 20: {pca_20.explained_variance_ratio_.sum():.4f}")
    print(f"PCA 50: {pca_50.explained_variance_ratio_.sum():.4f}")
    print(f"PCA 95%% retained components: {pca_var.n_components_}")
    print(f"PCA 95%% variance sum: {pca_var.explained_variance_ratio_.sum():.4f}")

    # ----------------------------
    # Save outputs
    # ----------------------------
    pd.DataFrame(X_all_scaled).to_csv(os.path.join(OUTPUT_DIR, "X_all_genes.csv"), index=False)
    pd.DataFrame(X_top).to_csv(os.path.join(OUTPUT_DIR, "X_top_variable_genes.csv"), index=False)
    pd.DataFrame(X_pca_20).to_csv(os.path.join(OUTPUT_DIR, "X_pca_20.csv"), index=False)
    pd.DataFrame(X_pca_50).to_csv(os.path.join(OUTPUT_DIR, "X_pca_50.csv"), index=False)
    pd.DataFrame(X_pca_var).to_csv(os.path.join(OUTPUT_DIR, "X_pca_95_var.csv"), index=False)
    pd.DataFrame(X_clinical_only).to_csv(os.path.join(OUTPUT_DIR, "X_clinical.csv"), index=False)
    y.to_csv(os.path.join(OUTPUT_DIR, "y_labels.csv"), index=False)

    # Save transformers
    joblib.dump(scaler_genes, os.path.join(OUTPUT_DIR, "scaler_genes.pkl"))
    joblib.dump(scaler_clinical, os.path.join(OUTPUT_DIR, "scaler_clinical.pkl"))
    joblib.dump(pca_20, os.path.join(OUTPUT_DIR, "pca_20.pkl"))
    joblib.dump(pca_50, os.path.join(OUTPUT_DIR, "pca_50.pkl"))
    joblib.dump(pca_var, os.path.join(OUTPUT_DIR, "pca_95_var.pkl"))

    # ----------------------------
    # Save summary + plots
    # ----------------------------
    summary = {
        "Total Samples": int(len(y)),
        "Number of Gene Features": int(X_genes.shape[1]),
        "Number of Clinical Features": int(X_clinical.shape[1]),
        "Number of Combined Features": int(X_all.shape[1]),
        "Positive Class Count": int((y == 1).sum()),
        "Negative Class Count": int((y == 0).sum()),
    }

    with open(os.path.join(OUTPUT_DIR, "dataset_summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    plt.figure()
    y.value_counts().sort_index().plot(kind="bar")
    plt.title("Class Distribution: Overall Survival")
    plt.xlabel("Label (0/1)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()

    plt.figure()
    cumulative_variance = np.cumsum(pca_var.explained_variance_ratio_)
    plt.plot(cumulative_variance, marker="o")
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_explained_variance.png"))
    plt.close()

    print("\nSaved processed files to:", OUTPUT_DIR)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()