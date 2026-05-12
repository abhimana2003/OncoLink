import streamlit as st
from auth import login_physician, register_physician


def render() :
    if st.session_state.get("_do_rerun"):
        del st.session_state["_do_rerun"]
        st.rerun()
        return

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
    .auth-logo { text-align: center; margin-bottom: 2.2rem; }
    .auth-logo .logo-icon { font-size: 3.2rem; display: block; margin-bottom: 0.4rem; }
    .auth-logo .logo-name {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 2.1rem; color: #1a3a6b; letter-spacing: -0.5px; display: block;
    }
    .auth-logo .logo-sub {
        font-family: 'DM Sans', sans-serif; font-size: 0.88rem; color: #7a8aab;
        letter-spacing: 1.5px; text-transform: uppercase; display: block; margin-top: 2px;
    }
    .auth-title {
        font-family: 'DM Serif Display', serif; font-size: 1.35rem;
        color: #1a3a6b; margin-bottom: 1.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2.2, 1])
    with center:
        st.markdown("""
        <div class='auth-logo'>
            <span class='logo-icon'>🩺</span>
            <span class='logo-name'>OncoLink</span>
            <span class='logo-sub'>Precision Medicine Platform</span>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            _login_form()

        with tab_register:
            _register_form()


def _set_auth(token, physician) :
    st.session_state["auth_token"] = token
    st.session_state["physician"] = physician
    st.session_state.pop("current_evaluation", None)
    st.session_state["_do_rerun"] = True


def _login_form():
    error_key = "login_error"

    with st.form("login_form", clear_on_submit=False):
        st.markdown("<div class='auth-title'>Welcome back</div>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="your.username")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In", width="stretch", type="primary")

    if err := st.session_state.pop(error_key, None):
        st.error(err)

    if submitted:
        if not username or not password:
            st.session_state[error_key] = "Please enter both username and password."
            st.rerun()
            return
        result = login_physician(username, password)
        if result["success"]:
            _set_auth(result["token"], result["physician"])
            st.rerun()
        else:
            st.session_state[error_key] = result["error"]
            st.rerun()


def _register_form():
    error_key = "register_error"
    success_key = "register_success"

    with st.form("register_form", clear_on_submit=False):
        st.markdown("<div class='auth-title'>Create your account</div>", unsafe_allow_html=True)
        full_name = st.text_input("Full Name", placeholder="Dr. Jane Smith")
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", placeholder="jsmith")
        with col2:
            specialty = st.text_input("Specialty", placeholder="Oncology")
        email = st.text_input("Email", placeholder="jane.smith@hospital.org")
        password = st.text_input("Password", type="password", placeholder="Min. 6 characters")
        confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
        submitted = st.form_submit_button("Create Account", width="stretch", type="primary")

    if err := st.session_state.pop(error_key, None):
        st.error(err)
    if msg := st.session_state.pop(success_key, None):
        st.success(msg)

    if submitted:
        if password != confirm:
            st.session_state[error_key] = "Passwords do not match."
            st.rerun()
            return
        if not full_name.strip() or not username.strip() or not email.strip():
            st.session_state[error_key] = "Full name, username, and email are all required."
            st.rerun()
            return
        result = register_physician(username, email, full_name, password, specialty)
        if result["success"]:
            login_result = login_physician(username, password)
            if login_result["success"]:
                _set_auth(login_result["token"], login_result["physician"])
                st.rerun()
            else:
                st.session_state[success_key] = "Account created! Please sign in on the Sign In tab."
                st.rerun()
        else:
            st.session_state[error_key] = result["error"]
            st.rerun()
