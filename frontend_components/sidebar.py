import streamlit as st
from auth import logout


def render(raw_df) :
    physician = st.session_state.get("physician", {})

    st.sidebar.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
        [data-testid="stSidebar"] { background-color: #0f1f3d; }
        [data-testid="stSidebar"] * { color: #cdd6f4 !important; }
        [data-testid="stSidebar"] .stSelectbox div,
        [data-testid="stSidebar"] .stSelectbox span,
        [data-testid="stSidebar"] div[role="combobox"] *,
        [data-testid="stSidebar"] button span { color: white !important; }
        [data-testid="stSidebar"] .stRadio label { color: #cdd6f4 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style='text-align:center; padding: 0.8rem 0 0.5rem 0;'>
            <span style='font-size:2rem;'>🩺</span>
            <div style='font-family:"DM Serif Display",Georgia,serif; font-size:1.3rem; font-weight:700;
                 color:#7eb8f7; letter-spacing:0px;'>OncoLink</div>
            <div style='font-size:0.72rem; color:#8fa3c8; margin-top:2px; letter-spacing:1px;'>
                PRECISION MEDICINE PLATFORM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Physician identity
    if physician:
        full = physician.get("full_name", "Physician")
        initials = "".join(w[0].upper() for w in full.split()[:2])
        specialty = physician.get("specialty") or "Oncology"
        st.sidebar.markdown(
            f"""
            <div style='background:rgba(126,184,247,0.1); border:1px solid rgba(126,184,247,0.25);
                 border-radius:10px; padding:0.65rem 0.85rem; margin:0.6rem 0 0.3rem 0;'>
                <div style='display:flex; align-items:center; gap:0.6rem;'>
                    <div style='width:32px; height:32px; background:#1a5276; border-radius:50%;
                         display:flex; align-items:center; justify-content:center;
                         font-size:0.75rem; font-weight:700; color:#7eb8f7; flex-shrink:0;'>
                        {initials}
                    </div>
                    <div>
                        <div style='font-size:0.85rem; font-weight:600; color:#cdd6f4;'>{full}</div>
                        <div style='font-size:0.72rem; color:#8fa3c8;'>{specialty}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "<div style='font-size:0.72rem; color:#8fa3c8; font-weight:600; letter-spacing:1.5px;"
        " margin-bottom:6px; text-transform:uppercase;'>Navigation</div>",
        unsafe_allow_html=True,
    )

    nav_options = {
        "🔬  Evaluate New Patient": "evaluate",
        "🗂️  My Patients": "patient_list",
        "📊  Model Insights": "analysis",
    }
    nav_label = st.sidebar.radio("nav", list(nav_options.keys()), label_visibility="collapsed")
    nav_page = nav_options[nav_label]

    st.sidebar.markdown("---")
    _render_status(raw_df)
    st.sidebar.markdown("---")

    if st.sidebar.button("Sign Out", width="stretch"):
        token = st.session_state.pop("auth_token", None)
        if token:
            logout(token)
        st.session_state.pop("physician", None)
        st.session_state.pop("current_evaluation", None)
        st.rerun()

    return nav_page


def _render_status(raw_df):
    from frontend_utils.data_loader import load_best_model, load_model_comparison

    try:
        from frontend_utils.agent_bridge import is_groq_available
        groq_ok = is_groq_available()
    except Exception:
        groq_ok = False

    st.sidebar.markdown(
        "<div style='font-size:0.72rem; color:#8fa3c8; font-weight:600; letter-spacing:1.5px;"
        " margin-bottom:4px; text-transform:uppercase;'>System Status</div>",
        unsafe_allow_html=True,
    )

    checks = [
        ("Historical Dataset", raw_df is not None),
        ("Trained Model", load_best_model() is not None),
        ("Model Results", load_model_comparison() is not None),
        ("Similarity Index", _check_sim()),
        ("Groq AI", groq_ok),
    ]
    for label, ok in checks:
        icon = "🟢" if ok else "🔴"
        st.sidebar.caption(f"{icon} {label}")


def _check_sim() :
    try:
        from similarity_engine import get_engine
        return get_engine().index is not None
    except Exception:
        return False
