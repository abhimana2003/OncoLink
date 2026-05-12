# OncoLink - Precision Medicine Platform
The goal of OncoLink is to support precision medicine by helping oncologists predict a patient’s response to a treatment plan and make more informed clinical decisions. OncoLink can predict, with probability/confidence, whether a patient will respond to treatment, show the top k most similar historical patients from the METABRIC dataset, allow physicians to enter outcomes for patients so we can improve the model, and let them see insights into how the model makes predictions.

## Genomics 2 Group (202605-3)

- Anjali Bhimanadham — apb2192
- Mahsa Mohajeri — mm6859
- Daniyah Taimur — drt2145

## Project Structure

```text
OncoLink_v4/
├── README.md
├── requirements.txt
├── run_project.sh
├── app.py
├── login_page.py
├── auth.py
├── evaluate_patient.py
├── patient_list.py
├── processing.py
├── model.py
├── similarity_engine.py
├── agent.py
├── outcome_store.py
├── incremental_learner.py
├── frontend_components/
│   ├── sidebar.py
│   ├── analysis_tabs.py
│   ├── evaluate_patient.py
│   └── submit_outcome.py
├── frontend_utils/
│   ├── data_loader.py
│   ├── agent_bridge.py
│   └── new_patient_store.py
├── data/
│   └── METABRIC_RNA_Mutation.csv
├── test data/
│   ├── README.txt
│   ├── manifest.csv
│   └── uploaded patient-file examples
└── outputs_metabric/
    ├── processed feature CSVs, scalers, PCA objects, charts, and similarity index
    ├── oncolink.db
    ├── new_patient_evaluations.csv
    ├── new_patient_feature_store.csv
    ├── outcome_feedback.csv
    ├── retrain_checkpoint.txt
    └── model_results/
        ├── best_model.pkl
        ├── best_model_info.txt
        ├── incremental_model.pkl
        ├── incremental_update_log.json
        ├── model_comparison.csv
        ├── shap_values.npy
        ├── shap_feature_names.json
        └── ROC/confusion/SHAP chart images
```


## Setup Instructions
### 1. Create a Groq API key

Create a Groq API key here:

```text
https://console.groq.com/keys
```

Groq is used for generated explanation text. The app can still run without a key, but explanation panels will use static fallback text.

### 2. Create the local environment file

Open a terminal in the project folder and copy the example environment file:

```bash
cp .env.example .env
```

Open the `.env` file that was just created and replace the placeholder value:

```bash
GROQ_API_KEY=your_groq_key_here
```

with your actual Groq API key. The other values in `.env.example` can stay as they are.

---

## How To Run the Full Pipeline and App

Run the project from the terminal:

```bash
./run_project.sh
```

This script handles the full setup and launch:

1. creates or reuses `.venv`
2. installs `requirements.txt`
3. runs `processing.py`
4. runs `model.py`
5. creates/updates local output files and database files under `outputs_metabric/`
6. launches Streamlit at `http://localhost:8501`

Once Streamlit starts, it should open the application in your browser. If it does not open automatically, go to:

```text
http://localhost:8501
```

To reset the local physician/session/patient database before launch:

```bash
./run_project.sh --reset-db
```

### Optional: Run each step manually

```bash
.venv/bin/python processing.py
.venv/bin/python model.py
.venv/bin/python -m streamlit run app.py
```

---

## Example Usage

1. Start the app with `./run_project.sh`
2. Open `http://localhost:8501`.
3. Create a physician account from the **Create Account** tab.
4. Sign in with the account you created.
5. Go to **Evaluate New Patient**.
6. Choose **Manual Entry** and enter de-identified clinical values, or choose **Attach Patient File** and upload one of the sample files from `test data/`.
7. Click **Evaluate New Patient**.
8. Review the prediction, explanation, and similar historical patients.
9. Save or revisit the evaluation from **My Patients**.
10. In **My Patients**, use **Update Outcome** when a confirmed outcome is available.
11. Go to **Model Insights** to inspect SHAP/feature importance, gene expression, and model comparison results.

Example uploaded-file workflow:

```text
test data/330_68_GSM615650.txt
```

Upload the file on **Evaluate New Patient**, select a patient record from the parsed records, adjust any prefilled clinical fields, and submit the evaluation.

---

## Reproducing Core Functionality and Results

To reproduce the main generated artifacts from the raw METABRIC CSV:

```bash
.venv/bin/python processing.py
.venv/bin/python model.py
```

`processing.py` should regenerate files such as:

- `outputs_metabric/X_all_genes.csv`
- `outputs_metabric/X_clinical.csv`
- `outputs_metabric/X_pca_20.csv`
- `outputs_metabric/X_pca_50.csv`
- `outputs_metabric/X_pca_95_var.csv`
- `outputs_metabric/y_labels.csv`
- `outputs_metabric/dataset_summary.txt`
- `outputs_metabric/class_distribution.png`
- `outputs_metabric/pca_explained_variance.png`

`model.py` should regenerate files such as:

- `outputs_metabric/model_results/best_model.pkl`
- `outputs_metabric/model_results/best_model_info.txt`
- `outputs_metabric/model_results/model_comparison.csv`
- `outputs_metabric/model_results/incremental_model.pkl`
- ROC and confusion matrix images
- `outputs_metabric/model_results/shap_values.npy`
- `outputs_metabric/model_results/shap_feature_names.json`
- `outputs_metabric/model_results/shap_summary.png`

Then run the app:

```bash
.venv/bin/python -m streamlit run app.py
```

---
## System Structure

```text
data/METABRIC_RNA_Mutation.csv
        |
        v
processing.py
  - cleans METABRIC columns
  - builds clinical, all-gene, top-gene, PCA-20, PCA-50, and PCA-95 feature files
  - saves scalers, PCA objects, charts, feature metadata, and dataset summary
        |
        v
model.py
  - trains Logistic Regression, Random Forest, optional XGBoost, and an SGD incremental baseline
  - writes model comparison, ROC/confusion plots, best_model.pkl, incremental_model.pkl, and SHAP/fallback artifacts
        |
        v
app.py
  - Streamlit entry point and authenticated navigation
        |
        +-- login_page.py                         -> sign in / create account
        +-- auth.py                               -> SQLite physicians, sessions, saved patients
        +-- evaluate_patient.py                   -> new-patient intake, prediction, explanation, similarity comparison
        +-- patient_list.py                       -> saved patients, edit/delete, confirmed outcomes
        +-- similarity_engine.py                  -> FAISS/sklearn patient similarity index
        +-- outcome_store.py                      -> outcome CSV and retrain threshold logic
        +-- incremental_learner.py                -> SGDClassifier partial_fit and optional model promotion
        +-- agent.py                              -> Groq/OpenAI-compatible explanation helper
        +-- frontend_components/sidebar.py        -> sidebar navigation and system status
        +-- frontend_components/analysis_tabs.py  -> SHAP, gene-expression, and model-comparison views
        +-- frontend_utils/data_loader.py         -> artifact loading, prediction helpers, upload parsing
        +-- frontend_utils/new_patient_store.py   -> new-patient evaluation and feature persistence
```


---
