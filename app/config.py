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
# Learning is opt-in; base model and dependencies are supplied by the server owner.
TRAIN_BASE = os.getenv('SIMS_TRAIN_BASE_MODEL_PATH', '')
TRAIN_DEVICE = os.getenv('SIMS_TRAIN_DEVICE', 'cuda')
TRAIN_CPU_MAX_PARAMS = int(os.getenv('SIMS_TRAIN_CPU_MAX_PARAMS', '500000000'))
TRAIN_MAX_CHUNKS = int(os.getenv('SIMS_TRAIN_MAX_CHUNKS', '200'))
TRAIN_MAX_STEPS = int(os.getenv('SIMS_TRAIN_MAX_STEPS', '100'))
TRAIN_TIMEOUT = int(os.getenv('SIMS_TRAIN_TIMEOUT_SECONDS', '14400'))
ADAPTER_URL = os.getenv('SIMS_ADAPTER_URL', 'http://127.0.0.1:8001').rstrip('/')
RERANK_MODEL = os.getenv('SIMS_RERANK_MODEL_PATH', '')
EMBED_QUERY_PREFIX = os.getenv('SIMS_EMBED_QUERY_PREFIX', 'query: ')
EMBED_PASSAGE_PREFIX = os.getenv('SIMS_EMBED_PASSAGE_PREFIX', 'passage: ')
