# OpinionMiner

A personalized content feed that learns what you like using Thompson Sampling (multi-armed bandit algorithm).

## What it does

Collects contrarian opinions from Reddit and various blogs, filters out low-quality content using an LLM, and personalizes your feed based on what you click or skip.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your Groq API key (free at console.groq.com)

# Initialize database
python -m src.database.db_manager

# Collect initial data
python -m src.data_collection.orchestrator

# Process the opinions
python process_opinions.py

# Run the app
streamlit run src/ui/app.py
```

Open http://localhost:8501 in your browser.

## How it works

The system uses a multi-armed bandit algorithm where each topic (technology, economics, philosophy, etc.) is an "arm". When you click on an opinion, that topic's arm gets rewarded. When you skip, it gets penalized. Over time, the feed learns to show you more of what you actually read.

Data comes from:
- Reddit RSS feeds (r/unpopularopinion, r/changemyview, etc.)
- Contrarian blogs (Astral Codex Ten, Marginal Revolution, etc.)
- Optional: HackerNews

Quality filtering and topic classification handled by Groq's LLM API (llama-3.3-70b).

## Tech stack

- Python 3.11+
- Streamlit for the UI
- SQLite for storage
- Groq API for LLM processing
- Sentence-BERT for embeddings
- Thompson Sampling for personalization

## License

MIT
