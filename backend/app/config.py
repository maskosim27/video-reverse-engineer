"""Env-based settings. Everything configurable via .env, no cloud required."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT = BACKEND_DIR.parent  # project root


def _load_dotenv() -> None:
    """Loader .env minimal (stdlib saja): KEY=VALUE, lewati komentar/baris kosong,
    strip quotes, tidak menimpa env yang sudah ada."""
    p = ROOT / ".env"
    if not p.is_file():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        pass


_load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _path(value: str, default: str) -> Path:
    v = value or default
    p = Path(v)
    return p if p.is_absolute() else ROOT / v.lstrip("./")


APP_PORT = int(_env("APP_PORT", "3000"))
BACKEND_PORT = int(_env("BACKEND_PORT", "8000"))

_db = _env("DATABASE_URL", "sqlite:///./data/app.db")
DB_PATH = _path(_db.replace("sqlite:///", ""), "./data/app.db")
VIDEO_DIR = _path(_env("VIDEO_STORAGE_PATH"), "./data/videos")
FRAME_DIR = _path(_env("FRAME_STORAGE_PATH"), "./data/frames")

AI_PROVIDER = _env("AI_PROVIDER", "openrouter")
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
NINEROUTER_API_KEY = _env("NINEROUTER_API_KEY", "")
# 9Router = gateway lokal OpenAI-compatible; default port 20128
NINEROUTER_BASE_URL = _env("NINEROUTER_BASE_URL", "http://127.0.0.1:20128/v1")
AI_MODEL = _env("AI_MODEL", "google/gemini-2.5-flash")
AI_TEMPERATURE = float(_env("AI_TEMPERATURE", "0.2"))
AI_MAX_TOKENS = int(_env("AI_MAX_TOKENS", "8000"))

# faster-whisper: model diunduh otomatis saat pertama dipakai (base ~145MB)
WHISPER_MODEL = _env("WHISPER_MODEL", "base")
OCR_LANG = _env("OCR_LANG", "eng+ind")

# Used to encrypt stored API keys. If unset, an ephemeral key is generated
# (keys won't survive restart) and a warning is logged.
SECRET_KEY = _env("SECRET_KEY", "")

for d in (DB_PATH.parent, VIDEO_DIR, FRAME_DIR):
    d.mkdir(parents=True, exist_ok=True)
