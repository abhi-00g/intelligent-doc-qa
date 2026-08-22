"""
Central configuration — loads .env and defines paths and defaults.
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- Environment ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Cost Dashboard SDK ---
# Sends telemetry to the AI Cost Dashboard backend.
# Graceful fallback: if the SDK isn't installed or the env vars
# aren't set, the RAG app works exactly as before.
#
# Set these in your .env or Streamlit Cloud secrets:
#   COST_DASHBOARD_API_KEY=lcd_abc123...
#   COST_DASHBOARD_ENDPOINT=https://your-render-app.onrender.com/api/v1

cost_tracker = None

_dashboard_api_key = os.getenv("COST_DASHBOARD_API_KEY", "")
_dashboard_endpoint = os.getenv("COST_DASHBOARD_ENDPOINT", "")

if _dashboard_api_key and _dashboard_endpoint:
    try:
        from llm_cost_sdk import CostTracker

        cost_tracker = CostTracker(
            api_key=_dashboard_api_key,
            endpoint=_dashboard_endpoint,
            batch_size=5,       # Flush every 5 events (RAG apps have low volume)
            flush_interval=10.0,  # Or every 10 seconds, whichever comes first
        )
        logging.getLogger(__name__).info(
            "Cost Dashboard SDK initialized — sending telemetry to %s",
            _dashboard_endpoint,
        )
    except ImportError:
        logging.getLogger(__name__).warning(
            "llm-cost-sdk not installed — dashboard telemetry disabled. "
            "Install with: pip install 'llm-cost-sdk @ git+https://github.com/abhi-00g/ai-cost-dashboard.git@main#subdirectory=sdk'"
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Failed to initialize Cost Dashboard SDK: %s — telemetry disabled", e
        )
else:
    logging.getLogger(__name__).info(
        "Cost Dashboard env vars not set — telemetry disabled. "
        "Set COST_DASHBOARD_API_KEY and COST_DASHBOARD_ENDPOINT to enable."
    )
