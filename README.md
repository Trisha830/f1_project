# Apex Pulse — F1 Community Intelligence

A Neuralink-inspired (not copied) dark, editorial-style dashboard for your Formula 1 comment-topic-modelling project. The notebook's workflow includes Reddit/YouTube comment collection, preprocessing, LDA, BERTopic, and optional Gemini-based topic labels.

## Run locally
1. Install Python 3.10+.
2. In a terminal, from this folder:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```
3. Open http://127.0.0.1:8000

## Connect your existing notebook output
Create a `data` folder and place any output CSV in it. Supported files, in preference order:
- `f1_comments_with_llm_labels.csv`
- `f1_comments_with_topics.csv`
- `f1_comments_cleaned.csv`
- `f1_youtube_comments.csv`
- `f1_reddit_comments.csv`

The API detects common comment/topic/sentiment/race column names. Or use **Import CSV** in the dashboard; it saves a normalized copy as `data/uploaded_comments.csv` (upload becomes the active dataset).

## Backend API
- `GET /api/health` — service status
- `GET /api/dashboard` — counts, topics, sentiment distribution, races, sample comments
- `POST /api/upload` — multipart CSV upload
- `POST /api/analyze` — keyword-based demo classifier for one comment

The single-comment analyzer is explicitly a baseline, not a trained ML model. For actual inference, connect your existing LDA/BERTopic model artifacts and Gemini topic-labeling pipeline. Do not commit API keys; use environment variables for credentials.

## Notes
- If no project CSV is present, the dashboard uses clearly labelled illustrative demo comments.
- This is a starter implementation; production deployment should add authentication, rate limits, validation, and privacy controls.
# f1_project
