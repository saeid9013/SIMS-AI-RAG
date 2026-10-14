import os
from pathlib import Path

DATA = Path(os.getenv('SIMS_DATA_DIR', 'data')).resolve()
DB = DATA / 'sims.sqlite3'
UPLOADS = DATA / 'uploads'
MAX_UPLOAD = int(os.getenv('SIMS_MAX_UPLOAD_MB', '40')) * 1024 * 1024
MAX_PAGES = int(os.getenv('SIMS_MAX_PAGES', '1000'))
MAX_TEXT = 8_000_000
COOKIE_SECURE = os.getenv('SIMS_COOKIE_SECURE', 'false').lower() == 'true'
EMBED_MODEL = os.getenv('SIMS_EMBED_MODEL_PATH', '')
ONLINE_URL = os.getenv('SIMS_ONLINE_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
ONLINE_KEY = os.getenv('SIMS_ONLINE_API_KEY', '')
OLLAMA_URL = os.getenv('SIMS_OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
