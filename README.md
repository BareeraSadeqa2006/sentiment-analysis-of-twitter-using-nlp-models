# Sentiment Analysis of Twitter Data using NLP Models

A small Flask web app that classifies tweets as **Positive / Negative / Neutral** using the
HuggingFace transformer model
[`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment).

## Features
- User registration + login (SQLite, hashed passwords)
- Tweet input → sentiment label + confidence score
- Preprocessing: lowercasing, URL/mention cleanup, emoji handling (`😊` → `happy`), slang expansion (`lol` → `laughing`, `ngl` → `honestly`)
- Soft light UI theme with Poppins font

## Project structure
```
app.py
templates/
  login.html
  register.html
  index.html
static/
  style.css
requirements.txt
```

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The app will:
1. Download the transformer model on first run (a few hundred MB).
2. Create `users.db` (SQLite) for accounts.
3. Serve on http://localhost:5000

Register an account, log in, then paste a tweet into the main page and click **Analyze Sentiment**.

## Sample tweets to try
- `she is so adorable 😊`
- `this is trash lol 🤮`
- `it's okay i guess 😐`

## Label mapping
| Model output | Meaning  |
|--------------|----------|
| `LABEL_0`    | Negative |
| `LABEL_1`    | Neutral  |
| `LABEL_2`    | Positive |
