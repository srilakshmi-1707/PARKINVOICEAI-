from flask import Flask, render_template, request, redirect, session
import sqlite3
import os
import joblib
import numpy as np
import uuid
import re
import librosa
from datetime import datetime
from audio_features import extract_features

# ---------------- App Setup ----------------
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.secret_key = "parkinvoice_secret"

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "parkinson_model.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- Load ML Model ----------------
model = joblib.load(MODEL_PATH)

# ---------------- Language Dictionary ----------------


LANG_TEXT = {

"English": {
    "app_name": "ParkinVoiceAI",
    "welcome_title": "Your Parkinson's Voice Analysis Assistant",
    "description": "Analyze your voice to assess Parkinson's risk quickly and easily.",
    "disclaimer_title": "Disclaimer",
    "disclaimer_text": "ParkinVoiceAI is an AI-based tool and does not replace professional medical advice.",
    "proceed": "Proceed",

    "signup": "Sign Up",
    "login": "Login",
    "already_have_account": "Already have an account?",
    "name": "Name",
    "age": "Age",
    "gender": "Gender",
    "select_gender": "Select Gender",
    "male": "Male",
    "female": "Female",
    "other": "Other",
    "phone": "Phone Number",
    "invalid_phone_alert": "Please enter a valid 10-digit Indian mobile number.",

    "dashboard_title": "Dashboard",
    "record_audio": "Record Audio",
    "upload_audio": "Upload Audio",
    "upload_description": "Select an audio file for Parkinson’s risk analysis",
    "predict_risk": "Predict Risk",
    "selected_file": "Selected File",
    "file_selected": "File Selected",
    "voice_recording": "Voice Recording",
    "record_description": "Record your voice for Parkinson’s risk analysis",
    "record_voice": "Record Voice",
    "recording": "Recording",
    "click_start": "Click start to begin",
    "start": "Start",
    "stop": "Stop",
    "predict_risk": "Predict Risk",
    "recording_progress": "Recording...",
    "recording_completed": "Recording completed",
    "mic_permission_denied": "Microphone permission denied",
     "upload_failed": "Upload failed",
     "upload_description": "Select an audio file for Parkinson’s risk analysis",
     "choose_file": "Choose Audio File",
"file_selected": "File Selected",
"selected_file": "Selected File",
"predict_risk": "Predict Risk",
"mic_permission_denied": "Microphone permission denied. Please allow access.",
"recording_progress": "Recording...",
"recording_completed": "Recording completed",
"upload_failed": "Upload failed. Try again.",
     "risk_prediction": "Risk Prediction",
     "voice_risk_analysis": "Your Voice Risk Analysis",
    "risk_level": "Risk Level",
    "previous_records": "Previous Records",
    "date": "Date",
    "risk_percent": "Risk %",
     "low_risk": "Low Risk. No immediate concern.",
        "medium_risk": "Moderate Risk. Regular monitoring recommended.",
        "high_risk": "High Risk. Please consult a neurologist.",
        "view_statistics": "View Statistics",
    "back_dashboard": "Back to Dashboard",
    "logout": "Logout"
},

"Tamil": {
    "app_name": "பார்கின்வாய்ஸ்AI",
    "welcome_title": "உங்கள் பார்கின்சன் குரல் பகுப்பாய்வு உதவியாளர்",
    "description": "உங்கள் குரலை பகுப்பாய்வு செய்து பார்கின்சன் அபாயத்தை எளிதாக மதிப்பிடுங்கள்.",
    "disclaimer_title": "மறுப்பு அறிவிப்பு",
    "disclaimer_text": "ParkinVoiceAI ஒரு AI கருவி மட்டுமே. இது மருத்துவ ஆலோசனையை மாற்றாது.",
    "proceed": "தொடரவும்",

    "signup": "பதிவு செய்",
    "login": "உள் நுழை",
    "already_have_account": "ஏற்கனவே கணக்கு உள்ளதா?",
    "name": "பெயர்",
    "age": "வயது",
    "gender": "பாலினம்",
    "select_gender": "பாலினத்தை தேர்ந்தெடுக்கவும்",
    "male": "ஆண்",
    "female": "பெண்",
    "other": "மற்றவை",
    "phone": "தொலைபேசி எண்",
    "invalid_phone_alert": "சரியான 10 இலக்க இந்திய மொபைல் எண்ணை உள்ளிடவும்.",

    "dashboard_title": "கட்டுப்பாட்டு பலகம்",
    "record_audio": "குரல் பதிவு",
    "upload_audio": "குரல் கோப்பை பதிவேற்று",
    "upload_description": "பார்கின்சன் அபாயத்தை மதிப்பிட ஒரு குரல் கோப்பை தேர்ந்தெடுக்கவும்",
    "predict_risk": "அபாயத்தை கணிக்கவும்",
    "selected_file": "தேர்ந்தெடுக்கப்பட்ட கோப்பு",
    "file_selected": "கோப்பு தேர்ந்தெடுக்கப்பட்டது",
    "upload_description": "பார்கின்சன் அபாயத்தை கணிக்க குரல் கோப்பை தேர்ந்தெடுக்கவும்",
"file_selected": "கோப்பு தேர்வு செய்யப்பட்டது",
"selected_file": "தேர்ந்தெடுக்கப்பட்ட கோப்பு",
"predict_risk": "அபாயத்தை கணிக்கவும்",
"mic_permission_denied": "மைக்ரோஃபோன் அனுமதி மறுக்கப்பட்டது",
"recording_progress": "பதிவு நடைபெறுகிறது...",
"recording_completed": "பதிவு முடிந்தது",
"upload_failed": "பதிவேற்றம் தோல்வியடைந்தது",
    "voice_recording": "குரல் பதிவு",
    "record_description": "பார்கின்சன் அபாயத்தை மதிப்பிட உங்கள் குரலை பதிவு செய்யவும்",
    "record_voice": "குரல் பதிவு செய்யவும்",
    "recording": "பதிவு நடைபெறுகிறது",
    "click_start": "தொடங்க Start அழுத்தவும்",
    "start": "தொடங்கு",
    "stop": "நிறுத்து",
    "predict_risk": "அபாயத்தை கணிக்கவும்",
    "recording_progress": "பதிவு நடைபெறுகிறது...",
    "recording_completed": "பதிவு முடிந்தது",
    "mic_permission_denied": "மைக்ரோஃபோன் அனுமதி மறுக்கப்பட்டது",
    "upload_failed": "பதிவேற்றம் தோல்வியடைந்தது",
    "choose_file": "கோப்பை தேர்வு செய்",
     "risk_prediction": "அபாய கணிப்பு",
    "voice_risk_analysis": "உங்கள் குரல் அபாய பகுப்பாய்வு",
    "risk_level": "அபாய நிலை",
    "previous_records": "முந்தைய பதிவுகள்",
    "date": "தேதி",
    "risk_percent": "அபாய %",
     "low_risk": "குறைந்த அபாயம். உடனடி கவலை இல்லை.",
        "medium_risk": "மிதமான அபாயம். வழக்கமான கண்காணிப்பு பரிந்துரைக்கப்படுகிறது.",
        "high_risk": "அதிக அபாயம். தயவுசெய்து நரம்பியல் மருத்துவரை அணுகவும்.",
    "back_dashboard": "டாஷ்போர்டுக்கு திரும்பவும்",
    "view_statistics": "புள்ளிவிவரங்களை காண்க",
    "logout": "வெளியேறு"
}
}

# ---------------- Database ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Create tables if not exists
db = get_db()

db.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    phone TEXT UNIQUE,
    language TEXT
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    risk_score REAL,
    created_at TEXT
)
""")

db.commit()
db.close()
def check_audio_quality(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
        y, _ = librosa.effects.trim(y, top_db=25)

        if len(y) < 1000:
            return "Poor"

        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        signal_power = np.mean(y ** 2)
        noise = y - librosa.effects.preemphasis(y)
        noise_power = np.mean(noise ** 2) + 1e-10

        snr = 10 * np.log10(signal_power / noise_power)

        print(f"RMS: {rms:.5f}, ZCR: {zcr:.5f}, SNR: {snr:.2f} dB")

        if rms < 0.005:
            return "Poor"
        if zcr > 0.15:
            return "Poor"
        if snr < 10:
            return "Poor"

        return "Good"

    except Exception as e:
        print("Quality check error:", e)
        return "Poor"


# ---------------- Utilities ----------------
def get_text():
    lang = session.get("language", "English")

    if lang not in LANG_TEXT:
        lang = "English"

    return LANG_TEXT[lang]


def is_valid_indian_phone(phone):
    return bool(re.fullmatch(r"[6-9]\d{9}", phone))


# ---------------- Routes ----------------

@app.route("/")
def welcome():
    return render_template(
        "welcome.html",
        text=get_text(),
        current_language=session.get("language", "English")
    )
def get_text():
    lang = session.get("language")

    if not lang:
        lang = "English"

    return LANG_TEXT.get(lang, LANG_TEXT["English"])

@app.route("/set_language", methods=["POST"])
def set_language():
    selected_language = request.form.get("language", "English")
    session["language"] = selected_language

    if "user_id" in session:
        db = get_db()
        db.execute(
            "UPDATE users SET language=? WHERE id=?",
            (selected_language, session["user_id"])
        )
        db.commit()
        db.close()

    return "", 204

@app.route("/signup", methods=["GET", "POST"])
def signup():
    text = get_text()

    if request.method == "POST":
        name = request.form.get("name")
        gender = request.form.get("gender")
        phone = request.form.get("phone")

        # ✅ Age validation
        try:
            age = int(request.form.get("age"))

            if age <= 0 or age > 120:
                return render_template(
                    "signup.html",
                    text=text,
                    error="Please enter a valid age"
                )

        except ValueError:
            return render_template(
                "signup.html",
                text=text,
                error="Age must be a number"
            )

        # ✅ Phone validation
        if not is_valid_indian_phone(phone):
            return render_template(
                "signup.html",
                text=text,
                error=text["invalid_phone_alert"]
            )

        db = get_db()

        existing_user = db.execute(
            "SELECT id FROM users WHERE phone=?",
            (phone,)
        ).fetchone()

        if existing_user:
            db.close()
            return render_template(
                "signup.html",
                text=text,
                error="Phone number already registered"
            )

        db.execute(
            "INSERT INTO users (name, age, gender, phone, language) VALUES (?,?,?,?,?)",
            (name, age, gender, phone, session.get("language", "English"))
        )

        db.commit()
        db.close()

        return redirect("/login")

    return render_template("signup.html", text=text)


@app.route("/login", methods=["GET", "POST"])
def login():
    text = get_text()

    if request.method == "POST":
        name = request.form["name"].strip()
        phone = request.form["phone"].strip()

        db = get_db()

        user = db.execute(
            "SELECT * FROM users WHERE phone=?",
            (phone,)
        ).fetchone()

        db.close()

        if user:
            session["name"] = user["name"]
            session["user_id"] = user["id"]

            # ✅ DO NOT TOUCH LANGUAGE HERE

            return redirect("/dashboard")
        else:
            return render_template(
                "login.html",
                text=text,
                error="Incorrect username or phone number"
            )

    return render_template("login.html", text=text)
@app.route("/dashboard")
def dashboard():
    if "name" not in session:
        return redirect("/login")

    db = get_db()

    # ✅ Fetch data
    rows = db.execute(
        "SELECT risk_score, created_at FROM predictions WHERE user_id=? ORDER BY created_at ASC LIMIT 10",
        (session["user_id"],)
    ).fetchall()

    # ✅ Convert to JSON-safe format
    for row in rows:
        rows.append([float(row[0]), row[1]])
    db.close()

    return render_template(
        "dashboard.html",
        text=get_text(),
        name=session["name"],
        history=rows,   # ✅ FIXED (comma added)
    )
@app.route("/record", methods=["GET", "POST"])
def record():
    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":

        if "audio" not in request.files:
            return "No audio file received", 400

        audio = request.files["audio"]

        if audio.filename == "":
            return "Empty file selected", 400

        filename = str(uuid.uuid4()) + ".wav"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio.save(filepath)
        # Check recording quality
        quality = check_audio_quality(filepath)
        print("Audio Quality:", quality)

        if quality == "Poor":
             print("Warning: Low quality audio")
        try:
            features = extract_features(filepath)
            probability = model.predict_proba(features)[0][1] * 100
        except Exception as e:
            return f"Audio processing error: {str(e)}", 500

        db = get_db()
        db.execute(
            "INSERT INTO predictions (user_id, risk_score, created_at) VALUES (?,?,?)",
            (session["user_id"], probability, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        db.commit()
        db.close()

        return redirect("/prediction")

    return render_template("record.html", text=get_text())

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":

        if "audio" not in request.files:
            return "No audio file received", 400

        audio = request.files["audio"]

        if audio.filename == "":
            return "No file selected", 400

        filename = str(uuid.uuid4()) + ".wav"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio.save(filepath)

        try:
            features = extract_features(filepath)
            probability = model.predict_proba(features)[0][1] * 100
        except Exception as e:
            return f"Audio processing error: {str(e)}"

        db = get_db()
        db.execute(
            "INSERT INTO predictions (user_id, risk_score, created_at) VALUES (?,?,?)",
            (session["user_id"], probability,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        db.commit()
        db.close()
        print("Prediction saved, redirecting...")

        return redirect("/prediction")

    return render_template("upload.html", text=get_text())
@app.route("/prediction")
def prediction():

    if "user_id" not in session:
        return redirect("/login")

    db = get_db()

    # Latest prediction
    result = db.execute(
        "SELECT risk_score FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
        (session["user_id"],)
    ).fetchone()

    if result:
        risk_score = round(float(result[0]), 2)
    else:
        risk_score = 0

    # Previous records
    rows = db.execute(
        "SELECT created_at, risk_score FROM predictions WHERE user_id=? ORDER BY created_at DESC",
        (session["user_id"],)
    ).fetchall()

    history = []

    for row in rows:
        history.append([row[0], row[1]])

    # Risk classification
    if risk_score < 30:
        risk_class = "low"
        alert_message = get_text()["low_risk"]

    elif risk_score < 60:
        risk_class = "medium"
        alert_message = get_text()["medium_risk"]

    else:
        risk_class = "high"
        alert_message = get_text()["high_risk"]

    db.close()

    return render_template(
        "prediction.html",
        risk_score=risk_score,
        risk_class=risk_class,
        alert_message=alert_message,
        history=history,
        text=get_text()
    )

@app.context_processor
def inject_text():
    return dict(text=get_text())


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug = False) 
        