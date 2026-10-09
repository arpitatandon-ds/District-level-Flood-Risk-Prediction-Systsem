"""Flood Early Warning System - Streamlit app (clean version).

Run:  streamlit run flood_app.py
Secrets (API keys, passwords) live in a .env file, never in this code.
"""
import hashlib
import hmac
import json
import os
import secrets
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

import gdown
import google.generativeai as genai
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
# Looks for the model in app/data first, then in the project-level data folder
MODEL_CANDIDATES = [
    BASE_DIR / "data" / "flood_model.pkl",
    BASE_DIR.parent / "data" / "flood_model.pkl",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])
USER_FILE = BASE_DIR / "users.json"

DRIVE_FILE_ID = os.getenv("MODEL_DRIVE_FILE_ID", "1HBe-_a-wAcymSXg8KDsfJmZVSIt3mBJf")
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Check the exact name with check_models.py; override it in .env if needed
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

RISK_LEVELS = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}
RISK_ICONS = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

st.set_page_config(page_title="Flood Early Warning System", page_icon="🌊", layout="wide")


# --------------------------------------------------------------------------
# Users (passwords are stored hashed, not as plain text)
# --------------------------------------------------------------------------
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return salt, digest


def new_user_record(password: str, role: str) -> dict:
    salt, digest = hash_password(password)
    return {"salt": salt, "hash": digest, "role": role}


def save_users(users: dict) -> None:
    USER_FILE.write_text(json.dumps(users, indent=2))


def load_users() -> dict:
    if USER_FILE.exists():
        try:
            return json.loads(USER_FILE.read_text())
        except json.JSONDecodeError:
            pass
    # First run: create an admin account. Set ADMIN_PASSWORD in .env.
    users = {"admin": new_user_record(os.getenv("ADMIN_PASSWORD", "admin123"), "admin")}
    save_users(users)
    return users


def verify_login(users: dict, username: str, password: str) -> bool:
    record = users.get(username)
    if not record:
        return False
    if "hash" in record:  # new format
        _, digest = hash_password(password, record["salt"])
        return hmac.compare_digest(digest, record["hash"])
    if "password" in record:  # old plain-text format: check, then upgrade it
        if hmac.compare_digest(record["password"], password):
            users[username] = new_user_record(password, record.get("role", "public"))
            save_users(users)
            return True
    return False


# --------------------------------------------------------------------------
# Model, prediction and alert helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading flood model...")
def load_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(DRIVE_URL, str(MODEL_PATH), quiet=True)
        except Exception as exc:
            raise FileNotFoundError(f"Could not download the model: {exc}") from exc
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_chatbot():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def to_risk_label(value: float) -> str:
    if value >= 0.7:
        return "High"
    if value >= 0.4:
        return "Medium"
    return "Low"


def predict_scores(model, df: pd.DataFrame):
    return model.predict(df[list(model.feature_names_in_)])


def send_alerts(message: str) -> list[str]:
    """Send SMS and/or email if credentials are set in .env. Returns status lines."""
    status = []

    sid, token = os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN")
    sms_from, sms_to = os.getenv("TWILIO_FROM"), os.getenv("ALERT_PHONE")
    if all([sid, token, sms_from, sms_to]):
        try:
            from twilio.rest import Client

            Client(sid, token).messages.create(body=message, from_=sms_from, to=sms_to)
            status.append("SMS sent")
        except Exception as exc:
            status.append(f"SMS failed: {exc}")
    else:
        status.append("SMS skipped (Twilio settings missing in .env)")

    mail_user, mail_pass = os.getenv("EMAIL_USER"), os.getenv("EMAIL_APP_PASSWORD")
    mail_to = os.getenv("ALERT_EMAIL")
    if all([mail_user, mail_pass, mail_to]):
        try:
            msg = MIMEText(message)
            msg["Subject"] = "Flood Alert"
            msg["From"], msg["To"] = mail_user, mail_to
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(mail_user, mail_pass)
                server.sendmail(mail_user, mail_to, msg.as_string())
            status.append("Email sent")
        except Exception as exc:
            status.append(f"Email failed: {exc}")
    else:
        status.append("Email skipped (email settings missing in .env)")
    return status


def risk_distribution_chart(labels: pd.Series):
    counts = labels.value_counts().reindex(RISK_LEVELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(counts.index, counts.values, color=[RISK_COLORS[r] for r in counts.index])
    ax.set_ylabel("Number of districts")
    ax.set_title("Flood Risk Distribution")
    return fig


def show_alert_status(lines: list[str]):
    for line in lines:
        (st.success if line.endswith("sent") else st.warning)(line)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("predictions", pd.DataFrame())
st.session_state.setdefault("role", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("chat_history", [])

users = load_users()

st.title("Flood Early Warning System 🌊")
st.header("District-Level Flood Risk Prediction")

# --------------------------------------------------------------------------
# Sidebar: login / register / logout
# --------------------------------------------------------------------------
st.sidebar.header("User Access")

if st.session_state.role:
    st.sidebar.success(f"Logged in as {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.role = None
        st.session_state.username = None
        st.rerun()
else:
    mode = st.sidebar.radio("Mode", ["Login", "Register"])

    if mode == "Login":
        with st.sidebar.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Login")
        if login_clicked:
            if verify_login(users, username, password):
                st.session_state.role = users[username]["role"]
                st.session_state.username = username
                st.rerun()
            else:
                st.sidebar.error("Incorrect username or password")
    else:
        with st.sidebar.form("register_form"):
            new_user = st.text_input("New username").strip()
            new_pass = st.text_input("New password (min 6 characters)", type="password")
            register_clicked = st.form_submit_button("Register")
        if register_clicked:
            if not new_user or len(new_pass) < 6:
                st.sidebar.error("Enter a username and a password of at least 6 characters.")
            elif new_user in users:
                st.sidebar.error("Username already exists.")
            else:
                users[new_user] = new_user_record(new_pass, "public")
                save_users(users)
                st.sidebar.success("Registered. Please log in.")

if not st.session_state.role:
    st.info("Please login or register from the sidebar to use the system.")
    st.stop()

# --------------------------------------------------------------------------
# Main app (logged in)
# --------------------------------------------------------------------------
try:
    model = load_model()
except Exception as exc:
    if MODEL_PATH.exists():
        st.error(
            f"Model file found at {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes) "
            "but it could not be loaded. It may be corrupted or saved with a "
            "different scikit-learn version."
        )
    else:
        st.error(
            f"Model file not found at: {MODEL_PATH}. Put flood_model.pkl in the "
            "'data' folder next to the app, or set the Drive file to 'Anyone with the link'."
        )
    st.exception(exc)
    st.stop()
features = list(model.feature_names_in_)
is_admin = st.session_state.role == "admin"

tab_single, tab_batch, tab_report, tab_chat = st.tabs(
    ["Single District", "Batch CSV", "Visualization Report", "Help Chatbot"]
)

# ---- Tab 1: single district ------------------------------------------------
with tab_single:
    st.subheader("Single District Prediction")
    with st.form("single_form"):
        values = {col: st.slider(col, 0.0, 1.0, 0.5) for col in features}
        district = st.text_input("District Name")
        lat = st.number_input("Latitude", value=20.5937)
        lon = st.number_input("Longitude", value=78.9629)
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not district.strip():
            st.warning("Please enter a district name.")
        else:
            row = pd.DataFrame(values, index=[0])
            score = float(predict_scores(model, row)[0])
            label = to_risk_label(score)

            record = row.assign(District=district.strip(), Latitude=lat, Longitude=lon,
                                RiskScore=round(score, 3), PredictedRisk=label)
            st.session_state.predictions = pd.concat(
                [st.session_state.predictions, record], ignore_index=True
            )

            st.subheader("Predicted Flood Risk")
            st.markdown(
                f"{RISK_ICONS[label]} **:{RISK_COLORS[label]}[{label}]** (score {score:.2f})"
            )

            if hasattr(model, "feature_importances_"):
                importance = pd.Series(model.feature_importances_, index=features).sort_values()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(importance.index, importance.values, color="teal")
                ax.set_title("Feature Importance")
                st.pyplot(fig)

            if is_admin and label == "High":
                st.session_state.pending_alert = f"High flood risk in {district.strip()}"

    if is_admin and st.session_state.get("pending_alert"):
        st.warning(st.session_state.pending_alert)
        if st.button("Send alert (SMS + email)"):
            show_alert_status(send_alerts(st.session_state.pop("pending_alert")))

# ---- Tab 2: batch CSV ------------------------------------------------------
with tab_batch:
    st.subheader("Batch CSV Prediction")
    if not is_admin:
        st.info("Only admins can upload a CSV.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            data = pd.read_csv(uploaded)
            missing = [c for c in features if c not in data.columns]
            if missing:
                st.error(f"CSV is missing columns: {missing}")
            else:
                data["PredictedRisk"] = pd.Series(predict_scores(model, data)).apply(to_risk_label)
                st.dataframe(data)
                st.pyplot(risk_distribution_chart(data["PredictedRisk"]))

                high = data[data["PredictedRisk"] == "High"]
                if not high.empty:
                    names = ", ".join(high["District"].astype(str)) if "District" in high else f"{len(high)} rows"
                    st.warning(f"High-risk: {names}")
                    if st.button("Send alert for high-risk districts"):
                        show_alert_status(send_alerts(f"High flood risk in: {names}"))

# ---- Tab 3: report ---------------------------------------------------------
with tab_report:
    st.subheader("Flood Prediction Report")
    report = st.session_state.predictions
    if report.empty:
        st.info("No predictions yet. Make a prediction in the first tab.")
    else:
        st.dataframe(report)
        st.pyplot(risk_distribution_chart(report["PredictedRisk"]))

        high = report[report["PredictedRisk"] == "High"]
        st.subheader("High-Risk Districts")
        if high.empty:
            st.write("No high-risk districts yet.")
        else:
            st.table(high[["District", "PredictedRisk"]])

        st.download_button(
            "Download report as CSV",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="flood_predictions_report.csv",
            mime="text/csv",
        )

# ---- Tab 4: chatbot --------------------------------------------------------
with tab_chat:
    st.subheader("Flood Prediction Chatbot 🤖")
    chatbot = get_chatbot()
    if chatbot is None:
        st.info("Chatbot is off: add GOOGLE_API_KEY to your .env file.")
    else:
        for msg in st.session_state.chat_history:
            st.markdown(f"**{msg['role']}:** {msg['text']}")

        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input("Ask about floods or using this system")
            sent = st.form_submit_button("Send")

        if sent and question.strip():
            st.session_state.chat_history.append({"role": "You", "text": question})
            try:
                prompt = (
                    "You are a helpful assistant for a flood early-warning app. "
                    "Answer briefly and clearly.\n\nQuestion: " + question
                )
                answer = chatbot.generate_content(prompt).text
            except Exception as exc:
                answer = f"Chatbot error: {exc}"
            st.session_state.chat_history.append({"role": "Bot", "text": answer})
            st.rerun()