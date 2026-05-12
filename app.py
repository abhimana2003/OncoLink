import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(
    page_title="OncoLink — Precision Medicine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0f1f3d; }
[data-testid="stSidebar"] * { color: #cdd6f4 !important; }
[data-testid="stSidebar"] .stSelectbox div,
[data-testid="stSidebar"] .stSelectbox span,
[data-testid="stSidebar"] div[role="combobox"] *,
[data-testid="stSidebar"] button span { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: #cdd6f4 !important; }

/* Main content */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none;
}
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #1a3a6b; }
.stMetric label { font-size: 0.82rem !important; color: #555 !important; }
.stMetric [data-testid="stMetricValue"] { font-size: 1.4rem !important; }

/* Patient card */
.patient-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
    border: 1px solid #c5d3f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.patient-card h3 { color: #1a3a6b; margin: 0 0 0.3rem 0; font-size: 1.1rem; }
.patient-card .sub { color: #556; font-size: 0.88rem; }

/* Section headers */
.section-header {
    border-left: 4px solid #1a5276;
    padding-left: 0.8rem;
    margin: 1.2rem 0 0.8rem 0;
    color: #1a3a6b;
    font-size: 1.05rem;
    font-weight: 600;
}

/* Prediction badge */
.pred-responder {
    display: inline-block;
    background: #d4edda; color: #155724;
    padding: 0.35rem 1rem; border-radius: 20px;
    font-weight: 700; font-size: 1rem;
    border: 1px solid #c3e6cb;
}
.pred-non-responder {
    display: inline-block;
    background: #f8d7da; color: #721c24;
    padding: 0.35rem 1rem; border-radius: 20px;
    font-weight: 700; font-size: 1rem;
    border: 1px solid #f5c6cb;
}

/* Similar patient rows */
.sim-row-responder { background-color: #f0fff4; border-left: 3px solid #38a169; }
.sim-row-non { background-color: #fff5f5; border-left: 3px solid #e53e3e; }

/* Disclaimer */
.disclaimer {
    background: #fff8e1; border: 1px solid #ffc107;
    border-radius: 8px; padding: 0.6rem 1rem;
    font-size: 0.82rem; color: #6d4c00; margin-bottom: 1rem;
}

/* Patient result cards */
.result-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, border-color 0.2s;
    position: relative;
}
.result-card:hover {
    box-shadow: 0 4px 16px rgba(26,58,107,0.10);
    border-color: #b0c4de;
}
.result-card .card-header-row {
    display: flex; justify-content: space-between; align-items: flex-start;
}
.result-card .card-title {
    font-size: 1.05rem; font-weight: 700; color: #1a3a6b;
    margin-bottom: 2px;
}
.result-card .card-icons {
    display: flex; gap: 0.35rem; flex-shrink: 0;
}
.result-card .card-date {
    font-size: 0.78rem; color: #8a9ab8; margin-bottom: 0.7rem;
}
.result-card .card-badges {
    display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.7rem;
}
.badge-resp {
    display: inline-block; background: #d4edda; color: #155724;
    padding: 0.2rem 0.7rem; border-radius: 14px; font-size: 0.78rem;
    font-weight: 600; border: 1px solid #c3e6cb;
}
.badge-non {
    display: inline-block; background: #f8d7da; color: #721c24;
    padding: 0.2rem 0.7rem; border-radius: 14px; font-size: 0.78rem;
    font-weight: 600; border: 1px solid #f5c6cb;
}
.badge-pending {
    display: inline-block; background: #fff3cd; color: #856404;
    padding: 0.2rem 0.7rem; border-radius: 14px; font-size: 0.78rem;
    font-weight: 600; border: 1px solid #ffc107;
}
.badge-confirmed {
    display: inline-block; background: #d1ecf1; color: #0c5460;
    padding: 0.2rem 0.7rem; border-radius: 14px; font-size: 0.78rem;
    font-weight: 600; border: 1px solid #bee5eb;
}
.card-detail-row {
    font-size: 0.85rem; color: #555; margin-bottom: 2px;
}
.card-detail-row strong { color: #333; }

/* Modal overlay for edit / delete */
.modal-overlay {
    background: linear-gradient(135deg, #f0f4ff, #e8f0fe);
    border: 1px solid #c5d3f0;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin: 0.5rem 0 1rem 0;
}
.modal-overlay .modal-title {
    font-weight: 700; font-size: 0.95rem; color: #1a3a6b;
    margin-bottom: 0.6rem;
}
.delete-confirm {
    background: #fff5f5; border: 1px solid #f5c6cb;
    border-radius: 12px; padding: 1.3rem 1.5rem;
    margin: 0.5rem 0 1rem 0;
}
.delete-confirm .modal-title {
    font-weight: 700; font-size: 0.95rem; color: #721c24;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


def _is_authenticated() :
    return bool(st.session_state.get("auth_token") and st.session_state.get("physician"))


def main():
    if not _is_authenticated():
        st.markdown("<style>[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)
        from login_page import render as render_login
        render_login()
        return

    from frontend_utils.data_loader import load_raw_data
    from frontend_utils.new_patient_store import NewPatientStore
    from frontend_components import sidebar as sidebar_mod
    import evaluate_patient
    import patient_list
    from frontend_components import analysis_tabs

    raw_df = load_raw_data()
    nav_page = sidebar_mod.render(raw_df)
    current_eval = st.session_state.get("current_evaluation")
    if current_eval is None:
        latest_id = NewPatientStore().latest_evaluation_id()
        if latest_id:
            latest = NewPatientStore().get_evaluation(latest_id)
            if latest is not None:
                st.session_state["current_evaluation"] = {
                    "evaluation_id": latest_id,
                    "profile": latest.get("profile", {}),
                    "prediction_result": latest.get("prediction", {}),
                }
                current_eval = st.session_state["current_evaluation"]


    if nav_page != "patient_list":
        st.session_state.pop("_view_patient_eval_id", None)
        st.session_state.pop("_outcome_modal_eval_id", None)
        st.session_state.pop("_edit_modal_eval_id", None)
        st.session_state.pop("_delete_modal_eval_id", None)

    if nav_page == "evaluate":
        evaluate_patient.render(raw_df)

    elif nav_page == "patient_list":
        patient_list.render()

    elif nav_page == "analysis":
        analysis_tabs.render(current_eval)


if __name__ == "__main__":
    main()
