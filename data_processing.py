import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import joblib
import urllib.request
import gzip
import shutil
import os

#Download GEO series matrix dataset
def download_geo(url, output_path):
    gz_path = output_path + ".gz"
    
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return
    
    urllib.request.urlretrieve(url, gz_path)
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"GEO Series Matrix saved to {output_path}")

#Extract clinical metadata
def parse_geo_clinical(filepath):
    clinical_data = {}
    sample_ids = []
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = line.strip().split("\t")[1:]
        
        elif line.startswith("!Sample_characteristics_ch1"):
            values = line.strip().split("\t")[1:]
            
            for i, val in enumerate(values):
                if i >= len(sample_ids):
                    continue
                
                sample = sample_ids[i]
                if sample not in clinical_data:
                    clinical_data[sample] = {}
                
                if ":" in val:
                    key, value = val.split(":", 1)
                    clinical_data[sample][key.strip()] = value.strip()
    
    clinical_df = pd.DataFrame.from_dict(clinical_data, orient="index")
    
    return clinical_df

# Load and merge datasets
geo_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25066/matrix/GSE25066_series_matrix.txt.gz"
geo_output = "data/GSE25066_series_matrix.txt"

os.makedirs("data", exist_ok=True)
download_geo(geo_url, geo_output)

expression_path = "data/cleaned_expression.csv"
labels_path = "data/labels.csv"

expression = pd.read_csv(expression_path, index_col=0)
labels = pd.read_csv(labels_path)

clinical_path = "data/GSE25066_series_matrix.txt"
clinical = parse_geo_clinical(clinical_path)
clinical = clinical.apply(lambda col: col.map(
    lambda x: x.replace('"', '').strip() if isinstance(x, str) else x
))
clinical.index = clinical.index.str.replace('"', '').str.strip()

clinical.columns = (
    clinical.columns
    .str.lower()
    .str.replace('"', '')
    .str.replace(' ', '_')
    .str.replace('-', '_')
    .str.strip()
)


#print("\nRaw Clinical Columns:")
#print(clinical.columns.tolist())

print("\n DATA INSPECTION")
print(f"Gene Expression Shape: {expression.shape}")
print("Rows represent patients; columns represent genes.\n")

print("\n Labels Preview:")
print(labels.head())

print("\n Clinical Data Preview:")
print(clinical.head())

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

# Clean clinical data
print("\n CLEANING CLINICAL DATA")

clinical.index.name = "Sample_ID"

clinical.columns = [col.lower().replace(" ", "_") for col in clinical.columns]

selected_cols = []

if "age_years" in clinical.columns:
    clinical["age_years"] = pd.to_numeric(clinical["age_years"], errors="coerce")
    selected_cols.append("age_years")

if "er_status_ihc" in clinical.columns:
    clinical["er_status_ihc"] = clinical["er_status_ihc"].map({
        "positive": 1, "negative": 0
    })
    selected_cols.append("er_status_ihc")

if "her2_status" in clinical.columns:
    clinical["her2_status"] = clinical["her2_status"].map({
        "positive": 1, "negative": 0
    })
    selected_cols.append("her2_status")

clinical = clinical[selected_cols]

clinical = clinical.fillna(clinical.mean())

print("Clinical Features Used:", clinical.columns.tolist())
print("Clinical Shape:", clinical.shape)

#print("Clinical columns:", clinical.columns.tolist())
#print("Data columns (last 10):", data.columns.tolist()[-10:])
#print("Clinical index sample:", clinical.index[:5].tolist())
#print("Data index sample:", data.index[:5].tolist())

clinical_reset = clinical.reset_index()          
clinical_reset.columns.values[0] = "Sample_ID"  

data_reset = data.reset_index()
data_reset.columns.values[0] = "Sample_ID"

data = pd.merge(data_reset, clinical_reset, on="Sample_ID", how="left")
data = data.set_index("Sample_ID")

clinical_columns = [col for col in clinical.columns if col in data.columns]
gene_columns = [col for col in expression.columns if col in data.columns]

print("Clinical columns found in data:", clinical_columns)
print("Data shape after merge:", data.shape)

# Clean data
print("\n DATA CLEANING")

data = data[~data.index.duplicated(keep="first")]

missing_values = data.isnull().sum().sum()
print(f"Total Missing Values: {missing_values}")

gene_columns = expression.columns.intersection(data.columns)
clinical_columns = clinical.columns.intersection(data.columns)

#gene_columns = data.columns.drop("Response")
data[gene_columns] = data[gene_columns].apply(
    pd.to_numeric, errors="coerce"
)

data[clinical_columns] = data[clinical_columns].fillna(0)

threshold = int(0.8 * len(data))
data = data.dropna(axis=1, thresh=threshold)

clinical_columns = [col for col in clinical.columns if col in data.columns]
gene_columns = gene_columns.intersection(data.columns)

data[gene_columns] = data[gene_columns].fillna(
    data[gene_columns].mean()
)

data[clinical_columns] = data[clinical_columns].apply(pd.to_numeric, errors="coerce")
data[clinical_columns] = data[clinical_columns].fillna(data[clinical_columns].mean())

#data.dropna(inplace=True)

print("Cleaned Dataset Shape:", data.shape)

X_genes = data[gene_columns]
X_clinical = data[clinical_columns]

X = data.drop(columns=["Response"])
y = data["Response"].astype(int)

print("\nFeature Shapes:")
print("Genes:", X_genes.shape)
print("Clinical:", X_clinical.shape)
print("Combined:", X.shape)

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


scaler_genes = StandardScaler()
X_genes_scaled = scaler_genes.fit_transform(X_genes)

scaler_clinical = StandardScaler()
X_clinical_scaled = scaler_clinical.fit_transform(X_clinical)

X_scaled = np.concatenate([X_genes_scaled, X_clinical_scaled], axis=1)

print("Feature standardization complete.")
print("Scaled Data Shape:", X_scaled.shape)

# Extract necessary features
print("\n FEATURE ENGINEERING")

X_all = X_scaled

num_top_genes = min(1000, X.shape[1])  
variances = np.var(X_genes_scaled, axis=0)
top_indices = np.argsort(variances)[-num_top_genes:]
X_top_genes = X_genes_scaled[:, top_indices]
X_top = np.concatenate([X_top_genes, X_clinical_scaled], axis=1)

print(f"Top Variable Genes Selected: {num_top_genes}")

# Principal Component Analysis (dimensionlaity reduction)

pca_20 = PCA(n_components=20, random_state=42)
pca_50 = PCA(n_components=50, random_state=42)
pca_var = PCA(n_components=0.95, random_state=42)

X_pca_20 = pca_20.fit_transform(X_genes_scaled)
X_pca_50 = pca_50.fit_transform(X_genes_scaled)
X_pca_var = pca_var.fit_transform(X_genes_scaled)

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


joblib.dump(scaler_genes, "outputs/scaler_genes.pkl")
joblib.dump(scaler_clinical, "outputs/scaler_clinical.pkl")
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