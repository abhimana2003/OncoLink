import csv
import json
import os
import uuid
from datetime import datetime
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs_metabric")
X_ALL_PATH = os.path.join(OUTPUTS_DIR, "X_all_genes.csv")
EVAL_META_PATH = os.path.join(OUTPUTS_DIR, "new_patient_evaluations.csv")
EVAL_FEATURE_PATH = os.path.join(OUTPUTS_DIR, "new_patient_feature_store.csv")

META_FIELDS = [
    "evaluation_id",
    "created_at",
    "patient_label",
    "source_type",
    "source_identifier",
    "sample_accession",
    "genomic_mode",
    "matched_gene_count",
    "profile_json",
    "prediction_json",
    "imputed_fields_json",
]


class NewPatientStore:
    """
    Stores evaluated new-patient records and their model-ready features
    """

    def __init__(self):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        self.feature_columns = self._load_feature_columns()
        self._ensure_meta_file()
        self._ensure_feature_file()

    def save_evaluation(
        self,
        patient_label,
        profile,
        prediction_result,
        all_features,
        source_type = "manual",
        source_identifier = "",
        sample_accession = "",
        genomic_mode = "cohort_mean_fallback",
        matched_gene_count = 0,
        imputed_fields = None,
    ) :
        evaluation_id = self._new_evaluation_id()

        meta_row = {
            "evaluation_id": evaluation_id,
            "created_at": datetime.now().isoformat(),
            "patient_label": patient_label,
            "source_type": source_type,
            "source_identifier": source_identifier,
            "sample_accession": sample_accession,
            "genomic_mode": genomic_mode,
            "matched_gene_count": int(matched_gene_count),
            "profile_json": json.dumps(profile),
            "prediction_json": json.dumps(prediction_result),
            "imputed_fields_json": json.dumps(imputed_fields or []),
        }

        with open(EVAL_META_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=META_FIELDS)
            writer.writerow(meta_row)

        feature_row = {"evaluation_id": evaluation_id}
        row_dict = all_features.iloc[0].to_dict()
        for col in self.feature_columns:
            feature_row[col] = row_dict.get(col, 0.0)

        with open(EVAL_FEATURE_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["evaluation_id"] + self.feature_columns)
            writer.writerow(feature_row)

        return evaluation_id

    def list_evaluations(self) :
        if not os.path.exists(EVAL_META_PATH):
            return pd.DataFrame(columns=META_FIELDS)
        df = pd.read_csv(EVAL_META_PATH)
        if df.empty:
            return df
        return df.sort_values("created_at", ascending=False).reset_index(drop=True)

    def get_evaluation(self, evaluation_id) :
        df = self.list_evaluations()
        if df.empty:
            return None
        match = df[df["evaluation_id"].astype(str) == str(evaluation_id)]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        row["profile"] = _safe_json_loads(row.get("profile_json", "{}"), {})
        row["prediction"] = _safe_json_loads(row.get("prediction_json", "{}"), {})
        row["imputed_fields"] = _safe_json_loads(row.get("imputed_fields_json", "[]"), [])
        return row

    def get_feature_rows(self, evaluation_ids) :
        return self.get_feature_rows_with_ids(evaluation_ids).drop(columns=["evaluation_id"], errors="ignore")

    def get_feature_rows_with_ids(self, evaluation_ids) :
        if not os.path.exists(EVAL_FEATURE_PATH):
            return pd.DataFrame(columns=["evaluation_id"] + self.feature_columns)
        df = pd.read_csv(EVAL_FEATURE_PATH)
        if df.empty:
            return pd.DataFrame(columns=["evaluation_id"] + self.feature_columns)
        eval_ids = [str(x) for x in evaluation_ids]
        match = df[df["evaluation_id"].astype(str).isin(eval_ids)].copy()
        if match.empty:
            return pd.DataFrame(columns=["evaluation_id"] + self.feature_columns)
        match["__order"] = match["evaluation_id"].astype(str).map({eid: i for i, eid in enumerate(eval_ids)})
        match = match.sort_values("__order")
        available = [c for c in ["evaluation_id"] + self.feature_columns if c in match.columns]
        return match[available]

    def latest_evaluation_id(self) :
        df = self.list_evaluations()
        if df.empty:
            return None
        return str(df.iloc[0]["evaluation_id"])

    def _ensure_meta_file(self):
        if os.path.exists(EVAL_META_PATH):
            return
        with open(EVAL_META_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=META_FIELDS)
            writer.writeheader()

    def _ensure_feature_file(self):
        if os.path.exists(EVAL_FEATURE_PATH):
            return
        with open(EVAL_FEATURE_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["evaluation_id"] + self.feature_columns)
            writer.writeheader()

    def _load_feature_columns(self) :
        if os.path.exists(X_ALL_PATH):
            return pd.read_csv(X_ALL_PATH, nrows=0).columns.tolist()
        return []

    def _new_evaluation_id(self) :
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"NP-{stamp}-{uuid.uuid4().hex[:6]}"


def _safe_json_loads(raw, default):
    try:
        return json.loads(raw)
    except Exception:
        return default
