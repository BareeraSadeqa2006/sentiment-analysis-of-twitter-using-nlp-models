"""
Sentiment Analysis of Twitter Data using NLP Models
Flask web app with user auth and a HuggingFace transformer backend.
Model: cardiffnlp/twitter-roberta-base-sentiment
"""

import os
import re
import sqlite3
from functools import wraps

import emoji
import torch
from flask import (
    Flask,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from werkzeug.security import check_password_hash, generate_password_hash

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-in-production")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

# ---------------------------------------------------------------------------
# Database (SQLite) for user auth
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        # init_db is a no-op if the schema already exists; safe to call on every
        # request and protects against the file being removed while running.
        init_db()
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Sentiment model (loaded once at startup)
# ---------------------------------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

print(f"Loading sentiment model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Common internet slang -> full words, helps the model understand short-form text
SLANG_DICT = {
    "lol": "laughing",
    "lmao": "laughing",
    "rofl": "laughing",
    "omg": "oh my god",
    "idk": "i do not know",
    "idc": "i do not care",
    "tbh": "to be honest",
    "ngl": "honestly",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "fyi": "for your information",
    "btw": "by the way",
    "brb": "be right back",
    "smh": "shaking my head",
    "thx": "thanks",
    "ty": "thank you",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "r": "are",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "dunno": "do not know",
    "nvm": "never mind",
    "bff": "best friend",
    "gg": "good game",
    "fr": "for real",
    "ikr": "i know right",
    "af": "a lot",
}


def replace_slang(text: str) -> str:
    """Replace slang tokens with their expanded meaning."""
    words = text.split()
    replaced = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(replaced)


def handle_emojis(text: str) -> str:
    """Convert emojis to their text description (e.g. 😊 -> ' happy ')."""
    # emoji.demojize converts 😊 -> :smiling_face_with_smiling_eyes:
    demojized = emoji.demojize(text, delimiters=(" ", " "))
    # Clean up the :word_word: style into readable words
    demojized = demojized.replace("_", " ")
    # Collapse any extra whitespace introduced
    return re.sub(r"\s+", " ", demojized).strip()


def clean_text(text: str) -> str:
    """Basic cleaning: urls, mentions, extra whitespace."""
    text = re.sub(r"http\S+|www\.\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "@user", text)           # normalize mentions
    text = re.sub(r"#(\w+)", r"\1", text)           # drop # but keep word
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    """Full preprocessing pipeline."""
    text = text.lower()
    text = handle_emojis(text)
    text = clean_text(text)
    text = replace_slang(text)
    return text


# ---------------------------------------------------------------------------
# Sentiment inference
# ---------------------------------------------------------------------------
def analyze_sentiment(text: str):
    """Run the transformer model and return (label, confidence, processed_text)."""
    processed = preprocess(text)
    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    label = LABEL_MAP[pred_idx]
    confidence = float(probs[pred_idx].item())
    return label, confidence, processed


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("index"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("register.html")

        db = get_db()
        existing = db.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            flash("That username is already taken.", "error")
            return render_template("register.html")

        db.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        db.commit()
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("index"))

        flash("Invalid username or password.", "error")
        return render_template("login.html")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/index", methods=["GET", "POST"])
@login_required
def index():
    result = None
    tweet = ""
    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()
        if tweet:
            label, confidence, processed = analyze_sentiment(tweet)
            result = {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "processed": processed,
            }
        else:
            flash("Please enter some text to analyze.", "error")

    return render_template(
        "index.html",
        username=session.get("username"),
        result=result,
        tweet=tweet,
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    init_db()
