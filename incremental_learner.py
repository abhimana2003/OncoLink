import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score


OUTPUTS_DIR = "outputs_metabric"
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "model_results")
INCREMENTAL_MODEL_PATH = os.path.join(RESULTS_DIR, "incremental_model.pkl")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pkl")
UPDATE_LOG_PATH = os.path.join(RESULTS_DIR, "incremental_update_log.json")
PROMOTE_THRESHOLD = 20


class IncrementalLearner:
    def __init__(self):
        self.model = None
        self._load_or_init()

    def update(self, X_new, y_new) :
        if self.model is None:
            return {"success": False, "error": "Incremental model not initialized"}

        if len(X_new) == 0:
            return {"success": False, "error": "No samples provided"}

        try:
            X_features = self._prepare_features(X_new)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        y_arr = np.array(y_new)

        # Measure AUC before update if we have enough data
        auc_before = None
        try:
            if len(X_features) >= 5 and len(np.unique(y_arr)) > 1:
                prob_before = self.model.predict_proba(X_features)[:, 1]
                auc_before = float(roc_auc_score(y_arr, prob_before))
        except Exception:
            pass
        try:
            self.model.partial_fit(X_features, y_arr, classes=[0, 1])
        except Exception as e:
            return {"success": False, "error": f"partial_fit failed: {e}"}
        auc_after = None
        try:
            if len(X_features) >= 5 and len(np.unique(y_arr)) > 1:
                prob_after = self.model.predict_proba(X_features)[:, 1]
                auc_after = float(roc_auc_score(y_arr, prob_after))
        except Exception:
            pass

        joblib.dump(self.model, INCREMENTAL_MODEL_PATH)
        self._log_update(len(X_features), auc_before, auc_after)
        promoted = self._maybe_promote()

        return {
            "success": True,
            "n_samples": len(X_features),
            "auc_before": round(auc_before, 4) if auc_before else None,
            "auc_after": round(auc_after, 4) if auc_after else None,
            "promoted": promoted,
        }

    def predict(self, X) :
        if self.model is None:
            return {"error": "Model not available"}
        try:
            X_features = self._prepare_features(X)
        except ValueError as e:
            return {"error": str(e)}
        pred = self.model.predict(X_features)
        prob = self.model.predict_proba(X_features)[:, 1]
        return {"predictions": pred.tolist(), "probabilities": prob.tolist()}

    def _prepare_features(self, X) :
        expected_names = getattr(self.model, "feature_names_in_", None)

        if expected_names is None:
            return X.values if hasattr(X, "values") else np.array(X)

        if hasattr(X, "columns"):
            missing = [name for name in expected_names if name not in X.columns]
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "..." if len(missing) > 5 else ""
                raise ValueError(f"Missing expected feature columns: {preview}{suffix}")
            return X.loc[:, list(expected_names)]

        arr = np.array(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(expected_names):
            raise ValueError(
                f"Expected {len(expected_names)} features, got {arr.shape[1]}"
            )
        return pd.DataFrame(arr, columns=expected_names)

    def get_update_history(self) :
        if not os.path.exists(UPDATE_LOG_PATH):
            return []
        try:
            with open(UPDATE_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            return []

    def total_samples_seen(self) :
        history = self.get_update_history()
        return sum(e.get("n_samples", 0) for e in history)

    def _load_or_init(self):
        if os.path.exists(INCREMENTAL_MODEL_PATH):
            try:
                self.model = joblib.load(INCREMENTAL_MODEL_PATH)
                return
            except Exception:
                pass

        if os.path.exists(BEST_MODEL_PATH):
            try:
                m = joblib.load(BEST_MODEL_PATH)
                if isinstance(m, SGDClassifier):
                    self.model = m
                    return
            except Exception:
                pass

        self._bootstrap_from_data()

    def _bootstrap_from_data(self):
        X_path = os.path.join(OUTPUTS_DIR, "X_all_genes.csv")
        y_path = os.path.join(OUTPUTS_DIR, "y_labels.csv")
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            self.model = None
            return
        try:
            X = pd.read_csv(X_path).values
            y = pd.read_csv(y_path).iloc[:, 0].values
            sgd = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000)
            sgd.fit(X, y)
            self.model = sgd
            os.makedirs(RESULTS_DIR, exist_ok=True)
            joblib.dump(self.model, INCREMENTAL_MODEL_PATH)
            print("IncrementalLearner: bootstrapped from full dataset.")
        except Exception as e:
            print(f"IncrementalLearner bootstrap failed: {e}")
            self.model = None

    def _maybe_promote(self) :
        if self.total_samples_seen() < PROMOTE_THRESHOLD:
            return False
        try:
            joblib.dump(self.model, BEST_MODEL_PATH)
            info = {
                "model_name": "SGD (Incremental)",
                "feature_name": "All Features",
                "promoted_at": datetime.now().isoformat(),
                "samples_used": self.total_samples_seen(),
            }
            with open(os.path.join(RESULTS_DIR, "best_model_info.txt"), "w") as f:
                f.write(str(info))
            return True
        except Exception:
            return False

    def _log_update(self, n, auc_before, auc_after):
        history = self.get_update_history()
        history.append({
            "timestamp": datetime.now().isoformat(),
            "n_samples": n,
            "auc_before": auc_before,
            "auc_after": auc_after,
        })
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(UPDATE_LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)
