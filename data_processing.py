import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import joblib

# Load and merge datasets
expression_path = "data/cleaned_expression.csv"
labels_path = "data/labels.csv"

expression = pd.read_csv(expression_path, index_col=0)
labels = pd.read_csv(labels_path)

print("\n DATA INSPECTION")
print(f"Gene Expression Shape: {expression.shape}")
print("Rows represent patients; columns represent genes.\n")

print("Labels Preview:")
print(labels.head())

labels.rename(columns={"Unnamed: 0": "Sample_ID"}, inplace=True)

labels.set_index("Sample_ID", inplace=True)

if "Response" in expression.columns:
    print("Warning: 'Response' column found in expression data. Dropping it.")
    expression = expression.drop(columns=["Response"])


data = expression.join(labels[["Response"]], how="inner")

print("\nMerged Dataset Shape:", data.shape)
print("Columns include Response:", "Response" in data.columns)

print("\nClass Distribution:")
print(data["Response"].value_counts())


# Clean data
print("\n DATA CLEANING")

data = data[~data.index.duplicated(keep="first")]

missing_values = data.isnull().sum().sum()
print(f"Total Missing Values: {missing_values}")


gene_columns = data.columns.drop("Response")
data[gene_columns] = data[gene_columns].apply(
    pd.to_numeric, errors="coerce"
)


threshold = int(0.8 * len(data))
data = data.dropna(axis=1, thresh=threshold)

gene_columns = data.columns.drop("Response")

data[gene_columns] = data[gene_columns].fillna(
    data[gene_columns].mean()
)

data.dropna(inplace=True)

print("Cleaned Dataset Shape:", data.shape)

X = data.drop(columns=["Response"])
y = data["Response"].astype(int)

print("\nFeature Matrix Shape:", X.shape)
print("Label Vector Shape:", y.shape)

# Dataset description
print("\n DATASET SUMMARY")

summary = {
    "Number of Patients": data.shape[0],
    "Number of Genes": data.shape[1] - 1,
    "Number of Features (after cleaning)": X.shape[1],
    "Number of Responders": int(y.sum()),
    "Number of Non-Responders": int((y == 0).sum())
}

for key, value in summary.items():
    print(f"{key}: {value}")


os.makedirs("outputs", exist_ok=True)

with open("outputs/dataset_summary.txt", "w") as f:
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print("Dataset summary saved to outputs/dataset_summary.txt")

# Standardizing data
print("\n PREPROCESSING")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature standardization complete.")
print("Scaled Data Shape:", X_scaled.shape)

# Extract necessary features
print("\n FEATURE ENGINEERING")

X_all = X_scaled

num_top_genes = min(1000, X.shape[1])  # Adjust if needed
variances = np.var(X_scaled, axis=0)
top_indices = np.argsort(variances)[-num_top_genes:]
X_top = X_scaled[:, top_indices]

print(f"Top Variable Genes Selected: {num_top_genes}")

# Principal Component Analysis (dimensionlaity reduction)

pca_20 = PCA(n_components=20, random_state=42)
pca_50 = PCA(n_components=50, random_state=42)
pca_var = PCA(n_components=0.95, random_state=42)

X_pca_20 = pca_20.fit_transform(X_scaled)
X_pca_50 = pca_50.fit_transform(X_scaled)
X_pca_var = pca_var.fit_transform(X_scaled)

print("PCA feature sets created.")

print("\nExplained Variance Ratios:")
print(f"PCA 20: {pca_20.explained_variance_ratio_.sum():.4f}")
print(f"PCA 50: {pca_50.explained_variance_ratio_.sum():.4f}")
print("Number of components retained:", pca_var.n_components_)
print("Explained variance:", pca_var.explained_variance_ratio_.sum())

# Split data for training and testing
print("\n TRAIN TEST SPLIT")

def stratified_split(X_features, y_labels, test_size=0.2):
    return train_test_split(
        X_features,
        y_labels,
        test_size=test_size,
        random_state=42,
        stratify=y_labels
    )

splits = {
    "all": stratified_split(X_all, y),
    "top": stratified_split(X_top, y),
    "pca_20": stratified_split(X_pca_20, y),
    "pca_50": stratified_split(X_pca_50, y),
    "pca_95_var": stratified_split(X_pca_var, y),
}

print("Stratified splits completed.")

# Create class distribution plot
plt.figure()
y.value_counts().sort_index().plot(kind="bar")
plt.title("Class Distribution of Treatment Response")
plt.xlabel("Response (0 = Non-Responder, 1 = Responder)")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.close()

print("Class distribution plot saved to outputs/class_distribution.png")

# Create PCA plots
plt.figure()
cumulative_variance = np.cumsum(pca_var.explained_variance_ratio_)
plt.plot(cumulative_variance, marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/pca_explained_variance.png")
plt.close()

print("PCA explained variance plot saved to outputs/pca_explained_variance.png")

# Saving necessary outputs
os.makedirs("outputs", exist_ok=True)

pd.DataFrame(X_all).to_csv("outputs/X_all_genes.csv", index=False)
pd.DataFrame(X_top).to_csv("outputs/X_top_variable_genes.csv", index=False)
pd.DataFrame(X_pca_20).to_csv("outputs/X_pca_20.csv", index=False)
pd.DataFrame(X_pca_50).to_csv("outputs/X_pca_50.csv", index=False)
pd.DataFrame(X_pca_var).to_csv("outputs/X_pca_95_var.csv", index=False)
y.to_csv("outputs/y_labels.csv", index=False)

print("\nProcessed datasets saved in the 'outputs/' directory.")


joblib.dump(scaler, "outputs/scaler.pkl")
joblib.dump(pca_20, "outputs/pca_20.pkl")
joblib.dump(pca_50, "outputs/pca_50.pkl")
joblib.dump(pca_var, "outputs/pca_95_var.pkl")

print("Scaler and PCA models saved.")

print("\n DATA SUMMARY")
print(f"Total Samples: {len(y)}")
print(f"Total Genes: {X.shape[1]}")
print("Feature Sets Prepared:")
print("- All Genes")
print("- Top Variable Genes")
print("- PCA (20, 50, 95% variance)")
print("\nData preprocessing complete.")