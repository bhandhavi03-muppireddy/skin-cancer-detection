import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from datetime import datetime
import random

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SmartDermaTrace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], footer, #MainMenu {
    display: none !important;
}

html, body, [data-testid="stAppViewContainer"], .stApp, section.main {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    height: 100% !important;
    overflow-x: hidden !important;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

section.main > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

:root{
  --teal:#1bb3a8;
  --teal2:#52f2df;
  --teal3:#11978f;
  --text:#e9f6f3;
  --muted:#9ab4af;
  --shadow:0 12px 45px rgba(0,0,0,.28);
}

.stApp {
    background:
      radial-gradient(circle at 50% 15%, rgba(27,179,168,.28), transparent 26%),
      radial-gradient(circle at 50% 55%, rgba(27,179,168,.08), transparent 38%),
      linear-gradient(90deg, #001112 0%, #062120 50%, #001112 100%);
    color: var(--text);
    min-height: 100vh;
}

/* Buttons */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--teal), var(--teal3)) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 0.9rem 1rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 12px 30px rgba(27,179,168,.22) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #22c8bc, #13978f) !important;
    color: white !important;
}
.stButton > button * {
    color: white !important;
}

/* Inputs */
.stTextInput > div > div,
.stTextInput input {
    background: rgba(255,255,255,.06) !important;
    color: #f4fffc !important;
    border-radius: 16px !important;
}
.stTextInput input {
    border: 1px solid rgba(82,242,223,.16) !important;
    padding: 0.95rem 1rem !important;
}
.stTextInput input::placeholder {
    color: #9ab4af !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.03) !important;
    border: 2px dashed rgba(82,242,223,.28) !important;
    border-radius: 22px !important;
    padding: 2rem !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #dff8f3 !important;
}

h1, h2, h3, h4, h5, h6, p, label, div {
    color: inherit;
}

.page-shell {
    min-height: 100vh;
    width: 100%;
}

.page-panel {
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 22px 40px;
    box-sizing: border-box;
}

.welcome-outer {
    width: 100%;
    padding-top: 28px;
    padding-bottom: 24px;
}

.welcome-box {
    width: 100%;
    max-width: 1180px;
    min-height: 620px;
    margin: 0 auto;
    border-radius: 42px;
    text-align: center;
    padding: 60px 40px 30px 40px;
    position: relative;
    overflow: hidden;
    background:
      radial-gradient(circle at center, rgba(27,179,168,.14), transparent 34%),
      linear-gradient(90deg, #001112 0%, #062120 50%, #001112 100%);
    border: 1px solid rgba(82,242,223,.08);
    box-sizing: border-box;
}

.welcome-box:before {
    content: "";
    position: absolute;
    width: 88%;
    height: 88%;
    top: -12%;
    left: 50%;
    transform: translateX(-50%);
    border-radius: 50%;
    border: 1px solid rgba(82,242,223,.08);
}

.welcome-inner {
    position: relative;
    z-index: 2;
    max-width: 900px;
    margin: 0 auto;
}

.welcome-eyebrow,
.auth-eyebrow,
.page-eyebrow {
    text-transform: uppercase;
    letter-spacing: .34em;
    font-size: .8rem;
    font-weight: 700;
    color: #3fd8cf;
    margin-bottom: 1.2rem;
}

.welcome-title {
    font-size: clamp(3rem, 7vw, 5.4rem);
    line-height: 1.04;
    font-weight: 800;
    margin-bottom: 1rem;
    color: #f6faf9;
}

.welcome-title em {
    color: #54f3e3;
    font-style: italic;
    font-weight: 700;
}

.welcome-divider {
    width: 72px;
    height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--teal2));
    margin: 0 auto 1.8rem auto;
    border-radius: 999px;
}

.welcome-sub {
    color: #8fa39f;
    font-size: 1.08rem;
    letter-spacing: .04em;
    margin-bottom: 1.3rem;
}

.welcome-note {
    color: #6c827d;
    font-size: .95rem;
    margin-top: 1rem;
}

.auth-wrap {
    width: 100%;
    padding-top: 36px;
    padding-bottom: 36px;
}

.auth-card-box {
    background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
    border: 1px solid rgba(82,242,223,.13);
    box-shadow: var(--shadow);
    border-radius: 30px;
    padding: 26px 24px 28px;
    backdrop-filter: blur(12px);
}

.auth-title {
    text-align: center;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    font-weight: 800;
    margin-bottom: .35rem;
    color: #f8fbfa;
}

.auth-sub {
    text-align: center;
    color: #819590;
    font-size: 1rem;
    margin-bottom: 1.6rem;
}

.field-label {
    font-size: .82rem;
    font-weight: 800;
    letter-spacing: .08em;
    color: #9aa7a4;
    text-transform: uppercase;
    margin: .5rem 0 .5rem;
}

.auth-footer {
    text-align: center;
    color: #7e908c;
    margin-top: 1.2rem;
    font-size: .98rem;
}

.mini-link {
    text-align: right;
    color: #9ab4af;
    font-size: .92rem;
    margin-top: -.2rem;
    margin-bottom: 1rem;
}

.social-row {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin: 1rem 0 1.2rem 0;
}

.social-icon {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(82,242,223,.14);
    color: #eafaf7;
    font-weight: 800;
    font-size: 1rem;
}

.page-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 22px 30px;
    border-bottom: 1px solid rgba(82,242,223,.08);
    background: rgba(0,0,0,.12);
    backdrop-filter: blur(8px);
    border-radius: 0 0 28px 28px;
    width: 100%;
    box-sizing: border-box;
}

.brand {
    color: #f2fbf8;
    font-size: 1.4rem;
    font-weight: 800;
}
.brand em {
    color: #57f1e4;
    font-style: italic;
}

.hero-panel {
    border-radius: 34px;
    padding: 36px;
    margin-bottom: 28px;
    background:
      radial-gradient(circle at center top, rgba(27,179,168,.18), transparent 35%),
      linear-gradient(90deg, #031515 0%, #0a2724 50%, #031515 100%);
    border: 1px solid rgba(82,242,223,.1);
    box-shadow: var(--shadow);
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    color: #f8fbfa;
    margin: 0 0 .5rem;
}
.hero-title em {
    color: #57f1e4;
    font-style: italic;
}

.hero-sub {
    color: #8ea49f;
    font-size: 1rem;
}

.section-label {
    color: #42d9cf;
    text-transform: uppercase;
    letter-spacing: .32em;
    font-size: .78rem;
    font-weight: 800;
    margin-bottom: 1rem;
}

.action-tile, .stat-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(82,242,223,.1);
    border-radius: 24px;
    padding: 24px;
    box-shadow: var(--shadow);
}
.action-tile {
    text-align: center;
}
.tile-icon {
    font-size: 1.9rem;
    margin-bottom: .7rem;
}
.tile-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f5fbf9;
    margin-bottom: .45rem;
}
.tile-sub {
    color: #8ea29e;
    font-size: .97rem;
}

.notice-card {
    margin-top: 20px;
    background: linear-gradient(135deg, rgba(27,179,168,.16), rgba(12,49,45,.8));
    border: 1px solid rgba(82,242,223,.16);
    border-radius: 22px;
    padding: 18px 20px;
    color: #dff7f2;
}

.page-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fbfa;
    margin: 0 0 .35rem;
}
.page-sub {
    color: #8fa49f;
    margin-bottom: 1.5rem;
    font-size: 1rem;
}

.upload-card, .report-card, .awareness-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(82,242,223,.1);
    border-radius: 26px;
    padding: 24px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}

.verdict-box {
    border-radius: 22px;
    padding: 1.4rem;
    text-align: center;
    margin-top: 1rem;
}
.verdict-danger {
    background: rgba(255,90,122,.08);
    border: 1px solid rgba(255,90,122,.25);
}
.verdict-safe {
    background: rgba(20,200,138,.08);
    border: 1px solid rgba(20,200,138,.24);
}
.verdict-icon {
    font-size: 2.2rem;
}
.verdict-label {
    font-size: 1.45rem;
    font-weight: 800;
    color: #f7fbfa;
    margin: .35rem 0;
}
.verdict-desc {
    color: #9bb0ab;
    line-height: 1.6;
}

.cancer-badge {
    text-align: center;
    margin-top: 1rem;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: .04em;
    padding: .9rem 1rem;
    border-radius: 16px;
}
.cancer-danger {
    background: rgba(255,90,122,.1);
    color: #ff839b;
    border: 1px solid rgba(255,90,122,.26);
}
.cancer-safe {
    background: rgba(20,200,138,.1);
    color: #38d99d;
    border: 1px solid rgba(20,200,138,.24);
}

.report-box {
    background: rgba(0,0,0,.18);
    border: 1px solid rgba(82,242,223,.1);
    border-left: 3px solid var(--teal);
    border-radius: 18px;
    font-family:'Courier New', monospace;
    font-size:.84rem;
    color:#e4f5f1;
    padding:1.2rem;
    white-space:pre-wrap;
    line-height:1.7;
}

.small-muted {
    color: #8ea29e;
    font-size: .95rem;
}

.stat-card {
    text-align: center;
}
.stat-num {
    font-size: 1.7rem;
    font-weight: 800;
    color: #56efe2;
}
.stat-lbl {
    font-size: .86rem;
    color: #9ab0ab;
    margin-top: .2rem;
}

.disclaimer {
    margin-top: 18px;
    border-radius: 18px;
    padding: 16px 18px;
    background: rgba(232,180,76,.08);
    border: 1px solid rgba(232,180,76,.18);
    color: #e8c778;
}

@media (max-width: 900px) {
    .welcome-box {
        min-height: auto;
        padding: 42px 20px 26px;
        margin: 12px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
@st.cache_resource
def get_model():
    try:
        return load_model("skin_cancer_model.h5", compile=False)
    except Exception:
        return None

model = get_model()

# =========================
# HELPERS
# =========================
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

def preprocess(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (128, 128))
    image = remove_hair(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def build_report():
    if not st.session_state.analysis_result:
        return ""
    r = st.session_state.analysis_result
    bar = '█' * round(r["conf"] / 100 * 28) + '░' * (28 - round(r["conf"] / 100 * 28))
    return "\n".join([
        "╔══════════════════════════════════════════════════════╗",
        "║          SmartDermaTrace — Analysis Report           ║",
        "╠══════════════════════════════════════════════════════╣",
        f"  Date & Time  : {r['date']}",
        f"  Patient      : {st.session_state.logged_user or 'User'}",
        f"  Model        : SkinCancer-CNN v1.0{' (Demo mode)' if r['is_demo'] else ''}",
        "──────────────────────────────────────────────────────",
        f"  RESULT       : {r['label']}",
        f"  STATUS       : {r['status']}",
        f"  Score        : {r['score']:.4f}",
        f"  Confidence   : [{bar}] {r['conf']}%",
        "──────────────────────────────────────────────────────",
        "  DISCLAIMER:",
        "  This tool is for screening purposes only and does",
        "  not constitute medical advice. Always consult a",
        "  certified dermatologist for a definitive diagnosis.",
        "╚══════════════════════════════════════════════════════╝",
    ])

def go(page):
    st.session_state.page = page

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "logged_user" not in st.session_state:
    st.session_state.logged_user = ""

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"

if "login_type" not in st.session_state:
    st.session_state.login_type = "Email"

# =========================
# WELCOME PAGE
# =========================
if st.session_state.page == "welcome":
    st.markdown("""
    <div class="welcome-outer">
        <div class="welcome-box">
            <div class="welcome-inner">
                <div class="welcome-eyebrow">AI-POWERED DERMATOLOGY</div>
                <div class="welcome-title">Smart<em>Derma</em>Trace</div>
                <div class="welcome-divider"></div>
                <div class="welcome-sub">Intelligent skin lesion screening at your fingertips</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2.2, 2.6, 2.2])
    with c2:
        if st.button("Continue →", key="continue_btn", use_container_width=True):
            go("auth")
            st.rerun()

    st.markdown(
        '<div class="welcome-note" style="text-align:center;">For informational use only · Not a substitute for medical advice</div>',
        unsafe_allow_html=True
    )

# =========================
# AUTH PAGE
# =========================
elif st.session_state.page == "auth":
    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)

    left, center, right = st.columns([1.2, 1.5, 1.2])

    with center:
        st.markdown('<div class="auth-card-box">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Login</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Access your SmartDermaTrace account</div>', unsafe_allow_html=True)

        mode1, mode2 = st.columns(2)
        with mode1:
            if st.button("Login", key="login_tab", use_container_width=True):
                st.session_state.auth_mode = "Login"
                st.rerun()
        with mode2:
            if st.button("Sign Up", key="signup_tab", use_container_width=True):
                st.session_state.auth_mode = "Sign Up"
                st.rerun()

        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        if st.session_state.auth_mode == "Login":
            st.markdown('<div class="field-label">Username / Email</div>', unsafe_allow_html=True)
            email = st.text_input(
                "Username / Email",
                placeholder="Type your username or email",
                label_visibility="collapsed",
                key="login_email"
            )

            st.markdown('<div class="field-label">Password</div>', unsafe_allow_html=True)
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Type your password",
                label_visibility="collapsed",
                key="login_password"
            )

            st.markdown('<div class="mini-link">Forgot password?</div>', unsafe_allow_html=True)

            if st.button("LOGIN", key="main_login_btn", use_container_width=True):
                if email and password:
                    st.session_state.logged_user = email.split("@")[0] if "@" in email else email
                    go("home")
                    st.rerun()
                else:
                    st.error("Please enter username/email and password.")

            st.markdown('<div style="text-align:center; color:#8fa49f; margin-top:18px;">Or Sign Up Using</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="social-row">
                <div class="social-icon">f</div>
                <div class="social-icon">t</div>
                <div class="social-icon">G</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="text-align:center; color:#8fa49f; margin-top:12px;">Or Sign Up Using</div>', unsafe_allow_html=True)
            sign1, sign2, sign3 = st.columns([1, 1.2, 1])
            with sign2:
                if st.button("SIGN UP", key="bottom_signup_btn", use_container_width=True):
                    st.session_state.auth_mode = "Sign Up"
                    st.rerun()

        else:
            st.markdown('<div class="auth-title" style="font-size:2.2rem; margin-top:4px;">Sign Up</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-sub">Create your SmartDermaTrace account</div>', unsafe_allow_html=True)

            st.markdown('<div class="field-label">Full Name</div>', unsafe_allow_html=True)
            name = st.text_input("Full Name", placeholder="Type your full name", label_visibility="collapsed", key="signup_name")

            st.markdown('<div class="field-label">Email</div>', unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="Type your email", label_visibility="collapsed", key="signup_email")

            st.markdown('<div class="field-label">Phone Number</div>', unsafe_allow_html=True)
            phone = st.text_input("Phone Number", placeholder="Type your phone number", label_visibility="collapsed", key="signup_phone")

            st.markdown('<div class="field-label">Password</div>', unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="Create your password", label_visibility="collapsed", key="signup_pw")

            if st.button("CREATE ACCOUNT", key="create_account_btn", use_container_width=True):
                if name and len(phone) == 10 and phone.isdigit() and "@" in email and "." in email and len(password) >= 6:
                    st.session_state.logged_user = name.split()[0]
                    go("home")
                    st.rerun()
                else:
                    st.error("Please fill all fields correctly.")

            st.markdown('<div class="auth-footer">Already have an account?</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns([1, 1.2, 1])
            with b2:
                if st.button("LOGIN", key="back_login_btn", use_container_width=True):
                    st.session_state.auth_mode = "Login"
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HOME PAGE
# =========================
elif st.session_state.page == "home":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="page-topbar">
        <div class="brand">Smart<em>Derma</em>Trace</div>
        <div style="color:#97aba7;font-weight:700">{st.session_state.logged_user}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-panel">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hero-panel">
        <div class="page-eyebrow">YOUR DASHBOARD</div>
        <div class="hero-title">Hello, <em>{st.session_state.logged_user or 'there'}</em> 👋</div>
        <div class="hero-sub">What would you like to do today?</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="action-tile"><div class="tile-icon">📷</div><div class="tile-title">Upload Image</div><div class="tile-sub">Analyse a skin lesion with AI</div></div>', unsafe_allow_html=True)
        if st.button("Open Upload", key="home_upload", use_container_width=True):
            go("upload")
            st.rerun()

    with c2:
        st.markdown('<div class="action-tile"><div class="tile-icon">📄</div><div class="tile-title">Download Report</div><div class="tile-sub">Get your analysis report</div></div>', unsafe_allow_html=True)
        if st.button("Open Report", key="home_report", use_container_width=True):
            go("report")
            st.rerun()

    with c3:
        st.markdown('<div class="action-tile"><div class="tile-icon">💡</div><div class="tile-title">Awareness</div><div class="tile-sub">Learn about skin health</div></div>', unsafe_allow_html=True)
        if st.button("Open Awareness", key="home_awareness", use_container_width=True):
            go("awareness")
            st.rerun()

    st.markdown("""
    <div class="notice-card">
        <b>ℹ️ Medical Disclaimer</b><br>
        This tool is for informational screening only. Results do not constitute medical diagnosis.
        Always consult a certified dermatologist.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# =========================
# UPLOAD PAGE
# =========================
elif st.session_state.page == "upload":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="page-topbar">
        <div class="brand">Smart<em>Derma</em>Trace</div>
        <div style="color:#97aba7;font-weight:700">{st.session_state.logged_user}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-panel">', unsafe_allow_html=True)

    back_col, _ = st.columns([1, 8])
    with back_col:
        if st.button("← Back", key="upload_back", use_container_width=True):
            go("home")
            st.rerun()

    st.markdown('<div class="page-eyebrow">AI ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Upload Skin Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a clear dermoscopic or close-up photo of the skin lesion for AI-powered analysis.</div>', unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded skin image", use_container_width=True)

        if st.button("🔬 Analyse Image", key="analyse_image", use_container_width=True):
            if model is None:
                pred = random.random()
                is_demo = True
            else:
                x = preprocess(img)
                pred = float(model.predict(x, verbose=0)[0][0])
                is_demo = False

            is_canc = pred > 0.5
            conf = round(pred * 100) if is_canc else round((1 - pred) * 100)
            label = "Possible Malignant Lesion" if is_canc else "Likely Benign"
            status = "Skin Cancer Detected" if is_canc else "Skin Cancer Not Detected"
            desc = (
                "The model detected features associated with malignant lesions. Please consult a dermatologist promptly for a clinical evaluation."
                if is_canc else
                "The model found no strong indicators of malignancy. Continue regular self-checks and annual skin screenings."
            )

            st.session_state.analysis_result = {
                "score": pred,
                "label": label,
                "status": status,
                "desc": desc,
                "conf": conf,
                "isCanc": is_canc,
                "date": datetime.now().strftime("%d %B %Y  %I:%M %p"),
                "is_demo": is_demo
            }
            st.rerun()

    if st.session_state.analysis_result:
        r = st.session_state.analysis_result
        verdict_class = "verdict-danger" if r["isCanc"] else "verdict-safe"
        verdict_icon = "⚠️" if r["isCanc"] else "✅"

        st.markdown(f"""
        <div class="verdict-box {verdict_class}">
            <div class="verdict-icon">{verdict_icon}</div>
            <div class="verdict-label">{r["label"]}</div>
            <div class="verdict-desc">{r["desc"]}</div>
        </div>
        """, unsafe_allow_html=True)

        if r["isCanc"]:
            st.markdown("""
            <div class="cancer-badge cancer-danger">
                🔴 SKIN CANCER DETECTED<br>
                <span style="font-size:.88rem;font-weight:600">Please consult a dermatologist immediately</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="cancer-badge cancer-safe">
                🟢 SKIN CANCER NOT DETECTED<br>
                <span style="font-size:.88rem;font-weight:600">Likely benign based on the uploaded image</span>
            </div>
            """, unsafe_allow_html=True)

        st.write(f"**Confidence:** {r['conf']}%")
        st.progress(r["conf"] / 100)
        st.info(f"Raw score: {r['score']:.4f} · Threshold: 0.5")

        if r["is_demo"]:
            st.warning("Running in demo mode because skin_cancer_model.h5 was not found.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📄 Save & View Report", key="save_report", use_container_width=True):
                go("report")
                st.rerun()
        with c2:
            if st.button("↺ Analyse Another Image", key="analyse_another", use_container_width=True):
                st.session_state.analysis_result = None
                st.rerun()

    st.markdown("""
    <div class="disclaimer">
        ⚠️ For informational use only. Does not constitute medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# REPORT PAGE
# =========================
elif st.session_state.page == "report":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="page-topbar">
        <div class="brand">Smart<em>Derma</em>Trace</div>
        <div style="color:#97aba7;font-weight:700">{st.session_state.logged_user}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-panel">', unsafe_allow_html=True)

    back_col, _ = st.columns([1, 8])
    with back_col:
        if st.button("← Back", key="report_back", use_container_width=True):
            go("home")
            st.rerun()

    st.markdown('<div class="page-eyebrow">YOUR ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Download Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Download a detailed text summary of your most recent skin analysis.</div>', unsafe_allow_html=True)

    report_text = build_report()
    if report_text:
        st.markdown("""
        <div class="report-card" style="text-align:center">
            <div style="font-size:2.1rem">📄</div>
            <h3 style="margin:.55rem 0;color:#f8fbfa">Analysis Report Ready</h3>
            <p class="small-muted">Your latest AI skin analysis is ready to download</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="report-box">{report_text}</div>', unsafe_allow_html=True)

        st.download_button(
            "⬇ Download Report (.txt)",
            data=report_text,
            file_name="SmartDermaTrace_Report.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="report-card" style="text-align:center">
            <div style="font-size:2.1rem">📭</div>
            <h3 style="margin:.55rem 0;color:#f8fbfa">No analysis report available yet</h3>
            <p class="small-muted">Upload and analyse a skin image first to generate your report.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Go to Upload →", key="goto_upload_report", use_container_width=True):
            go("upload")
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

# =========================
# AWARENESS PAGE
# =========================
elif st.session_state.page == "awareness":
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="page-topbar">
        <div class="brand">Smart<em>Derma</em>Trace</div>
        <div style="color:#97aba7;font-weight:700">{st.session_state.logged_user}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-panel">', unsafe_allow_html=True)

    back_col, _ = st.columns([1, 8])
    with back_col:
        if st.button("← Back", key="aware_back", use_container_width=True):
            go("home")
            st.rerun()

    st.markdown('<div class="page-eyebrow">EDUCATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Skin Health Awareness</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Learn how to protect your skin and detect warning signs early.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="awareness-card">
        <h3 style="margin:0 0 .5rem;color:#f8fbfa">🌞 Early detection saves lives</h3>
        <p class="small-muted">Skin cancer is one of the most common cancers worldwide, yet it's also one of the most treatable when caught early. Knowledge is your first line of defence.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat-card"><div class="stat-num">1 in 5</div><div class="stat-lbl">People may develop skin cancer</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><div class="stat-num">98%</div><div class="stat-lbl">High survival if caught early</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><div class="stat-num">+20%</div><div class="stat-lbl">Cases increasing over time</div></div>', unsafe_allow_html=True)

    st.markdown("### UV Protection")
    st.write("""
• Apply broad-spectrum SPF 30+ sunscreen daily  
• Reapply every 2 hours when outdoors  
• Wear protective clothing, hats, and sunglasses  
• Seek shade between 10 AM – 4 PM  
• Avoid tanning beds
""")

    st.markdown("### When to See a Doctor")
    st.write("""
• A mole or spot that bleeds without injury  
• A sore that doesn't heal  
• Sudden changes to an existing mole  
• Itching, tenderness, or pain in a lesion  
• A dark streak under a nail
""")

    st.markdown("""
    <div class="disclaimer">
        ⚠️ The information on this page is for educational purposes only. It is not a substitute for professional medical advice.
        Always consult a qualified dermatologist.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)
