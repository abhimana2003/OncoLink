import os
import sqlite3
import hashlib
import secrets
import json
from datetime import datetime
from contextlib import contextmanager
DB_PATH = os.path.join("outputs_metabric", "oncolink.db")
_DB_INITIALIZED = False

try:
    import bcrypt
    _BCRYPT = True
except ImportError:
    _BCRYPT = False



def _init_db():
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS physicians (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    UNIQUE NOT NULL,
                email       TEXT    UNIQUE NOT NULL,
                full_name   TEXT    NOT NULL,
                specialty   TEXT    DEFAULT '',
                pw_hash     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                last_login  TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token        TEXT PRIMARY KEY,
                physician_id INTEGER NOT NULL,
                created_at   TEXT NOT NULL,
                FOREIGN KEY (physician_id) REFERENCES physicians(id)
            );

            CREATE TABLE IF NOT EXISTS patients (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                physician_id    INTEGER NOT NULL,
                evaluation_id   TEXT    UNIQUE NOT NULL,
                patient_label   TEXT    NOT NULL,
                profile_json    TEXT    NOT NULL,
                prediction_json TEXT,
                true_outcome    INTEGER,
                outcome_notes   TEXT,
                outcome_at      TEXT,
                created_at      TEXT    NOT NULL,
                FOREIGN KEY (physician_id) REFERENCES physicians(id)
            );
        """)
        conn.commit()
        _DB_INITIALIZED = True
    finally:
        conn.close()


@contextmanager
def _conn():
    _init_db()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _hash_password(password) :
    if _BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def _check_password(password, pw_hash) :
    if _BCRYPT:
        try:
            return bcrypt.checkpw(password.encode(), pw_hash.encode())
        except Exception:
            pass
    if ":" in pw_hash:
        salt, hashed = pw_hash.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    return False


def register_physician(username, email, full_name,
                       password, specialty = "") :
    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}
    if not username.strip() or not email.strip() or not full_name.strip():
        return {"success": False, "error": "Username, email, and name are required."}
    try:
        with _conn() as conn:
            conn.execute(
                """INSERT INTO physicians
                   (username, email, full_name, specialty, pw_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (username.strip(), email.strip(), full_name.strip(),
                 specialty.strip(), _hash_password(password),
                 datetime.now().isoformat()),
            )
            row = conn.execute(
                "SELECT id FROM physicians WHERE username = ?", (username.strip(),)
            ).fetchone()
            return {"success": True, "physician_id": row["id"]}
    except sqlite3.IntegrityError as e:
        msg = str(e)
        if "username" in msg:
            return {"success": False, "error": "Username already taken."}
        if "email" in msg:
            return {"success": False, "error": "Email already registered."}
        return {"success": False, "error": "Registration failed."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def login_physician(username, password) :
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM physicians WHERE username = ?", (username.strip(),)
            ).fetchone()
            if row is None:
                return {"success": False, "error": "Username not found."}
            if not _check_password(password, row["pw_hash"]):
                return {"success": False, "error": "Incorrect password."}

            token = secrets.token_hex(32)
            conn.execute(
                "INSERT INTO sessions (token, physician_id, created_at) VALUES (?, ?, ?)",
                (token, row["id"], datetime.now().isoformat()),
            )
            conn.execute(
                "UPDATE physicians SET last_login = ? WHERE id = ?",
                (datetime.now().isoformat(), row["id"]),
            )
            return {
                "success": True,
                "token": token,
                "physician": {
                    "id": row["id"],
                    "username": row["username"],
                    "full_name": row["full_name"],
                    "email": row["email"],
                    "specialty": row["specialty"],
                },
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def logout(token):
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    except Exception:
        pass


def get_physician_by_token(token) :
    try:
        with _conn() as conn:
            row = conn.execute(
                """SELECT p.* FROM physicians p
                   JOIN sessions s ON s.physician_id = p.id
                   WHERE s.token = ?""",
                (token,),
            ).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


def save_patient(physician_id, evaluation_id, patient_label, profile, prediction = None) :
    try:
        with _conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO patients
                   (physician_id, evaluation_id, patient_label,
                    profile_json, prediction_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (physician_id, evaluation_id, patient_label,
                 json.dumps(profile),
                 json.dumps(prediction) if prediction else None,
                 datetime.now().isoformat()),
            )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_patients_for_physician(physician_id) :
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM patients WHERE physician_id = ? ORDER BY created_at DESC",
                (physician_id,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["profile"] = json.loads(d["profile_json"]) if d["profile_json"] else {}
                d["prediction"] = json.loads(d["prediction_json"]) if d["prediction_json"] else {}
                result.append(d)
            return result
    except Exception:
        return []


def get_patient_by_eval_id(evaluation_id) :
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE evaluation_id = ?", (evaluation_id,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["profile"] = json.loads(d["profile_json"]) if d["profile_json"] else {}
            d["prediction"] = json.loads(d["prediction_json"]) if d["prediction_json"] else {}
            return d
    except Exception:
        return None


def update_patient_outcome(evaluation_id, true_outcome, notes = "") :
    try:
        with _conn() as conn:
            conn.execute(
                """UPDATE patients SET true_outcome = ?, outcome_notes = ?, outcome_at = ?
                   WHERE evaluation_id = ?""",
                (true_outcome, notes, datetime.now().isoformat(), evaluation_id),
            )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_outcome_pending_count(physician_id) :
    try:
        with _conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as n FROM patients
                   WHERE physician_id = ? AND true_outcome IS NOT NULL""",
                (physician_id,),
            ).fetchone()
            return row["n"] if row else 0
    except Exception:
        return 0


def update_patient_prediction(evaluation_id, prediction):
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE patients SET prediction_json = ? WHERE evaluation_id = ?",
                (json.dumps(prediction), evaluation_id),
            )
    except Exception:
        pass


def update_patient_profile(evaluation_id, patient_label, profile) :
    try:
        with _conn() as conn:
            conn.execute(
                """UPDATE patients SET patient_label = ?, profile_json = ?
                   WHERE evaluation_id = ?""",
                (patient_label, json.dumps(profile), evaluation_id),
            )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_patient(evaluation_id) :
    try:
        with _conn() as conn:
            conn.execute(
                "DELETE FROM patients WHERE evaluation_id = ?",
                (evaluation_id,),
            )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

_init_db()
