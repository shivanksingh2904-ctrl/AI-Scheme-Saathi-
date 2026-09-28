"""Central settings. Everything configurable lives here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "schemes.json"
STATIC_DIR = BASE_DIR / "static"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "schemes"

TOP_K = 6  # how many chunks to retrieve per question
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.8"))  # cosine distance cut-off

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_TEMPERATURE = 0.2  # low = more factual, less creative
LLM_TIMEOUT = 30  # seconds


def llm_configured() -> bool:
    """True only if all three LLM settings are filled with real values."""
    placeholders = {"your_key_here", "your_model_name_here", "https://api.example.com/v1"}
    values = [LLM_API_KEY, LLM_BASE_URL, LLM_MODEL]
    return all(values) and not any(v in placeholders for v in values)
