import os
import csv
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from incremental_learner import IncrementalLearner
from frontend_utils.new_patient_store import NewPatientStore


OUTPUTS_DIR = "outputs_metabric"
FEEDBACK_CSV = os.path.join(OUTPUTS_DIR, "outcome_feedback.csv")

FEEDBACK_FIELDS = [
    "timestamp",
    "patient_index",
    "evaluation_id",
    "true_outcome",
    "predicted_outcome",
    "predicted_probability",
    "notes"
]

RETRAIN_THRESHOLD = 10


class OutcomeStore:
    """
    Manages physician-submitted outcome feedback and incremental retraining
    """

    def __init__(self):
        self._ensure_csv()
        self.learner = IncrementalLearner()

    def record_outcome(self, patient_index, true_outcome, predicted_outcome=None, predicted_probability=None, evaluation_id=None, notes=""):
        """
        Record a confirmed outcome and potentially trigger model update
        """

        timestamp = datetime.now().isoformat()

        row = {
            "timestamp": timestamp,
            "patient_index": patient_index if patient_index is not None else "",
            "evaluation_id": evaluation_id or "",
            "true_outcome": int(true_outcome),
            "predicted_outcome": predicted_outcome if predicted_outcome is not None else "",
            "predicted_probability": f"{predicted_probability:.4f}" if predicted_probability is not None else "",
            "notes": notes.replace("\n", " "),
        }

        try:
            with open(FEEDBACK_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
                writer.writerow(row)

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to write outcome: {e}",
                "retrained": False,
                "new_count": 0
            }

        pending = self._count_pending_since_last_retrain()
        retrained = False

        target_label = f"evaluation {evaluation_id}" if evaluation_id else f"patient #{patient_index}"

        message = f"Outcome recorded for {target_label}."

        if pending >= RETRAIN_THRESHOLD:
            retrain_result = self._trigger_retrain()
            retrained = retrain_result.get("success", False)

            if retrained:
                message += f" Model updated with {pending} new outcomes."
                self._mark_retrain_checkpoint()

            else:
                message += f" Retrain attempted but failed: {retrain_result.get('error', 'unknown')}"

        return {
            "success": True,
            "message": message,
            "retrained": retrained,
            "new_count": pending,
        }

    def get_all_feedback(self):
        """
        Return all recorded outcomes as a DataFrame
        """

        if not os.path.exists(FEEDBACK_CSV):
            return pd.DataFrame(columns=FEEDBACK_FIELDS)

        try:
            df = pd.read_csv(FEEDBACK_CSV)
            return df

        except Exception:
            return pd.DataFrame(columns=FEEDBACK_FIELDS)

    def get_feedback_stats(self):
        """
        Summary statistics over all recorded outcomes
        """

        df = self.get_all_feedback()

        if df.empty:
            return {
                "total": 0,
                "responders": 0,
                "non_responders": 0,
                "pending_retrain": 0,
                "retrain_threshold": RETRAIN_THRESHOLD,
            }

        total = len(df)
        responders = int((df["true_outcome"] == 1).sum())
        non_responders = int((df["true_outcome"] == 0).sum())
        pending = self._count_pending_since_last_retrain()

        return {
            "total": total,
            "responders": responders,
            "non_responders": non_responders,
            "pending_retrain": pending,
            "retrain_threshold": RETRAIN_THRESHOLD,
        }

    def _trigger_retrain(self):
        """
        Load all feedback and call incremental learner
        """

        df = self.get_all_feedback()

        if df.empty:
            return {"success": False, "error": "No feedback data"}

        X_path = os.path.join(OUTPUTS_DIR, "X_all_genes.csv")

        if not os.path.exists(X_path):
            return {"success": False, "error": "Feature data not found"}

        try:
            X_all = pd.read_csv(X_path)

            frames = []
            labels = []

            hist_rows = df[df["evaluation_id"].fillna("").astype(str).str.strip() == ""].copy()

            if not hist_rows.empty:
                hist_rows = hist_rows[hist_rows["patient_index"].astype(str).str.strip() != ""].copy()

                if not hist_rows.empty:
                    hist_rows["patient_index"] = hist_rows["patient_index"].astype(int)

                    max_idx = len(X_all) - 1

                    hist_rows = hist_rows[hist_rows["patient_index"] <= max_idx]

                    if not hist_rows.empty:
                        frames.append(
                            X_all.iloc[hist_rows["patient_index"].values].reset_index(drop=True)
                        )

                        labels.extend(hist_rows["true_outcome"].astype(int).tolist())

            ext_rows = df[df["evaluation_id"].fillna("").astype(str).str.strip() != ""].copy()

            if not ext_rows.empty:
                eval_ids = ext_rows["evaluation_id"].astype(str).tolist()

                ext_features = NewPatientStore().get_feature_rows_with_ids(eval_ids)

                if not ext_features.empty:
                    label_map = {
                        str(row["evaluation_id"]): int(row["true_outcome"])
                        for _, row in ext_rows.iterrows()
                    }

                    ext_features = ext_features.reset_index(drop=True)

                    aligned_labels = [
                        label_map[str(eid)]
                        for eid in ext_features["evaluation_id"].astype(str).tolist()
                        if str(eid) in label_map
                    ]

                    frames.append(
                        ext_features.drop(columns=["evaluation_id"], errors="ignore")
                    )

                    labels.extend(aligned_labels)

            if not frames:
                return {
                    "success": False,
                    "error": "No valid feedback rows with usable features"
                }

            X_new = pd.concat(frames, ignore_index=True)
            y_new = np.array(labels, dtype=int)

            result = self.learner.update(X_new, y_new)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _count_pending_since_last_retrain(self):
        checkpoint_path = os.path.join(OUTPUTS_DIR, "retrain_checkpoint.txt")

        df = self.get_all_feedback()

        if df.empty:
            return 0

        if not os.path.exists(checkpoint_path):
            return len(df)

        try:
            with open(checkpoint_path) as f:
                last_ts = f.read().strip()

            return int((df["timestamp"] > last_ts).sum())

        except Exception:
            return len(df)

    def _mark_retrain_checkpoint(self):
        checkpoint_path = os.path.join(OUTPUTS_DIR, "retrain_checkpoint.txt")

        with open(checkpoint_path, "w") as f:
            f.write(datetime.now().isoformat())

    def _ensure_csv(self):
        if not os.path.exists(FEEDBACK_CSV):
            with open(FEEDBACK_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
                writer.writeheader()