import os
import sys
from pathlib import Path

# Ensure project root (backend/) is on sys.path so `import app` works
THIS_DIR = Path(__file__).parent
BACKEND_ROOT = THIS_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Keep OpenAI disabled by default during tests
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("SHAP_ENABLED", "false")
