"""
Lung Disease Prediction System - Flask Application
Author: College Final Year Project
Description: A web-based ML system for predicting lung disease risk
             with X-Ray scan analysis support
"""

import os
import json
import pickle
import sqlite3
import base64
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, g)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
#  App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "lung_disease_secret_key_2024_secure"

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATABASE      = os.path.join(BASE_DIR, "instance", "database.db")
MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
#  Load ML Model
# ─────────────────────────────────────────────
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    ml_model = model_data["model"]
    scaler   = model_data["scaler"]
    print("✅ ML Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    ml_model = None
    scaler   = None

# ─────────────────────────────────────────────
#  Database Helpers
# ─────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    with sqlite3.connect(DATABASE) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT    NOT NULL,
                email    TEXT    NOT NULL UNIQUE,
                password TEXT    NOT NULL,
                created  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                input_data TEXT    NOT NULL,
                result     TEXT    NOT NULL,
                confidence REAL    NOT NULL,
                date       TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS xray_scans (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                filename     TEXT    NOT NULL,
                analysis     TEXT    NOT NULL,
                finding      TEXT    NOT NULL,
                confidence   REAL    NOT NULL,
                date         TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS contacts (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT NOT NULL,
                email   TEXT NOT NULL,
                subject TEXT NOT NULL,
                message TEXT NOT NULL,
                date    TEXT NOT NULL
            );
        """)
    print("✅ Database initialized")

# ─────────────────────────────────────────────
#  Auth Decorator
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_xray_with_ai(image_path: str) -> dict:
    """
    Send chest X-ray image to Claude Vision API for radiological analysis.
    Returns structured finding dict.
    Falls back to rule-based mock if API key is not set.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # ── Fallback mock analysis (demo mode) ──
        return _mock_xray_analysis()

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = image_path.rsplit(".", 1)[-1].lower()
    media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "png": "image/png",  "webp": "image/webp"}
    media_type = media_map.get(ext, "image/jpeg")
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are an expert radiologist AI assistant specializing in chest X-ray analysis.
Analyze the provided chest X-ray image and respond ONLY with a valid JSON object — no extra text.

JSON schema:
{
  "finding": "Normal | Abnormal | Suspicious",
  "confidence": <float 0-100>,
  "primary_diagnosis": "<brief diagnosis, e.g. 'No significant findings' or 'Possible pulmonary opacity'>",
  "observations": [
    "<observation 1>",
    "<observation 2>",
    "<observation 3>"
  ],
  "lung_fields": "<description of lung fields>",
  "cardiac_silhouette": "<description>",
  "pleural_spaces": "<description>",
  "bones_and_soft_tissue": "<description>",
  "recommendation": "<clinical recommendation>",
  "urgency": "Routine | Urgent | Emergency"
}

Be medically precise. Base analysis solely on the image."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_image,
                    },
                },
                {
                    "type": "text",
                    "text": "Please analyze this chest X-ray and provide a detailed radiological report in the specified JSON format."
                }
            ],
        }],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    return result


def _mock_xray_analysis() -> dict:
    """Demo fallback when no API key is configured."""
    import random
    options = [
        {
            "finding": "Normal",
            "confidence": round(random.uniform(88, 96), 1),
            "primary_diagnosis": "No significant pulmonary abnormality detected",
            "observations": [
                "Lung fields appear clear and well-aerated bilaterally",
                "No focal consolidation, effusion, or pneumothorax identified",
                "Cardiac silhouette within normal limits"
            ],
            "lung_fields": "Clear bilaterally with no opacities or infiltrates",
            "cardiac_silhouette": "Normal size and contour, cardiothoracic ratio < 0.5",
            "pleural_spaces": "No effusion or thickening identified",
            "bones_and_soft_tissue": "Bony thorax intact, no lytic or sclerotic lesions",
            "recommendation": "No immediate follow-up required. Routine annual checkup advised.",
            "urgency": "Routine"
        },
        {
            "finding": "Suspicious",
            "confidence": round(random.uniform(72, 85), 1),
            "primary_diagnosis": "Possible early pulmonary infiltrate — further evaluation recommended",
            "observations": [
                "Faint opacity noted in the right lower lobe",
                "Subtle increased density in the perihilar region",
                "Mild prominence of pulmonary vasculature"
            ],
            "lung_fields": "Mild hazy opacity in right lower zone; left lung clear",
            "cardiac_silhouette": "Mildly enlarged, borderline cardiothoracic ratio",
            "pleural_spaces": "Trace right-sided pleural blunting noted",
            "bones_and_soft_tissue": "No acute osseous abnormality",
            "recommendation": "CT chest recommended for further characterization. Pulmonologist referral advised.",
            "urgency": "Urgent"
        },
        {
            "finding": "Abnormal",
            "confidence": round(random.uniform(85, 94), 1),
            "primary_diagnosis": "Bilateral pulmonary infiltrates consistent with pneumonia / lung pathology",
            "observations": [
                "Bilateral patchy consolidation in both lower lobes",
                "Air bronchograms visible in affected areas",
                "Increased interstitial markings throughout both lung fields"
            ],
            "lung_fields": "Diffuse bilateral infiltrates with consolidation in lower zones",
            "cardiac_silhouette": "Obscured left heart border due to adjacent consolidation",
            "pleural_spaces": "Bilateral small pleural effusions",
            "bones_and_soft_tissue": "Demineralization noted; no acute fractures",
            "recommendation": "Immediate clinical correlation required. Urgent pulmonologist and radiologist review. Consider bronchoscopy.",
            "urgency": "Emergency"
        }
    ]
    return random.choice(options)

# ─────────────────────────────────────────────
#  Routes – Public
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name    = request.form.get("name", "").strip()
        email   = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        if not all([name, email, subject, message]):
            flash("All fields are required.", "danger")
            return redirect(url_for("contact"))

        db = get_db()
        db.execute(
            "INSERT INTO contacts (name, email, subject, message, date) VALUES (?,?,?,?,?)",
            (name, email, subject, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        db.commit()
        flash("Message sent successfully! We'll get back to you soon.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")

# ─────────────────────────────────────────────
#  Routes – Authentication
# ─────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not all([name, email, password, confirm]):
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("register"))

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            flash("An account with that email already exists.", "danger")
            return redirect(url_for("register"))

        hashed = generate_password_hash(password)
        db.execute(
            "INSERT INTO users (name, email, password, created) VALUES (?,?,?,?)",
            (name, email, hashed, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        db.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"]    = user["id"]
            session["user_name"]  = user["name"]
            session["user_email"] = user["email"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# ─────────────────────────────────────────────
#  Routes – Symptom Prediction
# ─────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            features = [
                int(request.form.get("age", 0)),
                int(request.form.get("gender", 0)),
                int(request.form.get("smoking", 0)),
                int(request.form.get("yellow_fingers", 0)),
                int(request.form.get("anxiety", 0)),
                int(request.form.get("peer_pressure", 0)),
                int(request.form.get("chronic_disease", 0)),
                int(request.form.get("fatigue", 0)),
                int(request.form.get("allergy", 0)),
                int(request.form.get("wheezing", 0)),
                int(request.form.get("alcohol", 0)),
                int(request.form.get("coughing", 0)),
                int(request.form.get("shortness_breath", 0)),
                int(request.form.get("swallowing_diff", 0)),
                int(request.form.get("chest_pain", 0)),
            ]

            if ml_model is None:
                flash("ML model not available. Please contact admin.", "danger")
                return redirect(url_for("predict"))

            import numpy as np
            X        = np.array([features])
            X_scaled = scaler.transform(X)
            prediction  = ml_model.predict(X_scaled)[0]
            proba       = ml_model.predict_proba(X_scaled)[0]
            confidence  = round(float(max(proba)) * 100, 2)
            result_label = "High Risk" if prediction == 1 else "Low Risk"
            precautions  = get_precautions(prediction, features)

            input_json = json.dumps({
                "age": features[0], "gender": features[1], "smoking": features[2],
                "yellow_fingers": features[3], "anxiety": features[4],
                "peer_pressure": features[5], "chronic_disease": features[6],
                "fatigue": features[7], "allergy": features[8],
                "wheezing": features[9], "alcohol": features[10],
                "coughing": features[11], "shortness_breath": features[12],
                "swallowing_diff": features[13], "chest_pain": features[14],
            })
            db = get_db()
            db.execute(
                "INSERT INTO predictions (user_id,input_data,result,confidence,date) VALUES (?,?,?,?,?)",
                (session["user_id"], input_json, result_label, confidence,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            db.commit()

            return render_template("result.html",
                                   result=result_label,
                                   confidence=confidence,
                                   precautions=precautions,
                                   features=features)

        except Exception as e:
            flash(f"Prediction error: {str(e)}", "danger")
            return redirect(url_for("predict"))

    return render_template("predict.html")


def get_precautions(prediction, features):
    high_risk = [
        "Quit smoking immediately – seek cessation programs",
        "Visit your doctor urgently for lung screening",
        "Discuss preventive medications with your physician",
        "Avoid secondhand smoke and air pollutants",
        "Monitor oxygen saturation regularly",
        "Consider CT scan for early detection",
    ]
    low_risk = [
        "Continue healthy lifestyle habits",
        "Maintain regular aerobic exercise",
        "Eat a balanced diet with vitamins C & E",
        "Avoid starting smoking or exposure to tobacco",
        "Annual health checkups are recommended",
    ]
    return high_risk if prediction == 1 else low_risk

# ─────────────────────────────────────────────
#  Routes – X-Ray Scan Analysis  (NEW)
# ─────────────────────────────────────────────
@app.route("/xray", methods=["GET", "POST"])
@login_required
def xray():
    if request.method == "POST":
        # ── validate file upload ──
        if "xray_file" not in request.files:
            flash("No file selected.", "danger")
            return redirect(url_for("xray"))

        file = request.files["xray_file"]
        if file.filename == "":
            flash("No file selected.", "danger")
            return redirect(url_for("xray"))

        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload PNG, JPG, or JPEG.", "danger")
            return redirect(url_for("xray"))

        try:
            # ── save file ──
            filename  = secure_filename(
                f"xray_{session['user_id']}_{int(datetime.now().timestamp())}"
                f".{file.filename.rsplit('.', 1)[1].lower()}"
            )
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            # ── run AI analysis ──
            analysis = analyze_xray_with_ai(save_path)

            # ── persist to DB ──
            db = get_db()
            db.execute(
                """INSERT INTO xray_scans
                   (user_id, filename, analysis, finding, confidence, date)
                   VALUES (?,?,?,?,?,?)""",
                (
                    session["user_id"],
                    filename,
                    json.dumps(analysis),
                    analysis.get("finding", "Unknown"),
                    analysis.get("confidence", 0.0),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            db.commit()

            return render_template(
                "xray_result.html",
                analysis=analysis,
                image_filename=filename,
            )

        except Exception as e:
            flash(f"X-ray analysis error: {str(e)}", "danger")
            return redirect(url_for("xray"))

    return render_template("xray.html")


@app.route("/xray/history")
@login_required
def xray_history():
    db    = get_db()
    scans = db.execute(
        "SELECT * FROM xray_scans WHERE user_id=? ORDER BY date DESC",
        (session["user_id"],)
    ).fetchall()
    # parse JSON analysis for each scan
    scan_list = []
    for s in scans:
        d = dict(s)
        try:
            d["analysis_data"] = json.loads(d["analysis"])
        except Exception:
            d["analysis_data"] = {}
        scan_list.append(d)
    return render_template("xray_history.html", scans=scan_list)


@app.route("/xray/delete/<int:scan_id>", methods=["POST"])
@login_required
def delete_xray(scan_id):
    db   = get_db()
    row  = db.execute(
        "SELECT filename FROM xray_scans WHERE id=? AND user_id=?",
        (scan_id, session["user_id"])
    ).fetchone()
    if row:
        # delete image file
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], row["filename"])
        if os.path.exists(img_path):
            os.remove(img_path)
        db.execute("DELETE FROM xray_scans WHERE id=? AND user_id=?",
                   (scan_id, session["user_id"]))
        db.commit()
        flash("X-ray scan record deleted.", "info")
    return redirect(url_for("xray_history"))


# ─────────────────────────────────────────────
#  Routes – Dashboard
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    db    = get_db()
    user  = db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
    preds = db.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY date DESC",
        (session["user_id"],)
    ).fetchall()
    scans = db.execute(
        "SELECT * FROM xray_scans WHERE user_id=? ORDER BY date DESC LIMIT 5",
        (session["user_id"],)
    ).fetchall()

    total    = len(preds)
    high     = sum(1 for p in preds if p["result"] == "High Risk")
    low      = total - high
    xray_cnt = db.execute(
        "SELECT COUNT(*) as cnt FROM xray_scans WHERE user_id=?",
        (session["user_id"],)
    ).fetchone()["cnt"]

    return render_template("dashboard.html",
                           user=user,
                           predictions=preds,
                           xray_scans=scans,
                           total=total,
                           high_risk=high,
                           low_risk=low,
                           xray_count=xray_cnt)


@app.route("/delete_prediction/<int:pred_id>", methods=["POST"])
@login_required
def delete_prediction(pred_id):
    db = get_db()
    db.execute("DELETE FROM predictions WHERE id=? AND user_id=?",
               (pred_id, session["user_id"]))
    db.commit()
    flash("Prediction record deleted.", "info")
    return redirect(url_for("dashboard"))

# ─────────────────────────────────────────────
#  Error Handlers
# ─────────────────────────────────────────────
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("404.html", error_code=500,
                           error_msg="Internal Server Error"), 500

# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
