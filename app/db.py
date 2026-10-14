import sqlite3
from contextlib import contextmanager
from .config import DATA, DB, UPLOADS

@contextmanager
def connect():
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    DATA.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
          role TEXT NOT NULL CHECK(role IN ('admin','user')), active INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS sessions (
          token TEXT PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
          csrf TEXT NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS documents (
          id INTEGER PRIMARY KEY, name TEXT NOT NULL, storage TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL,
          revision TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', error TEXT NOT NULL DEFAULT '',
          warnings TEXT NOT NULL DEFAULT '', created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          created_by INTEGER REFERENCES users(id), chunk_count INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS chunks (
          id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          location TEXT NOT NULL, text TEXT NOT NULL, embedding BLOB);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='unicode61');
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS history (
          id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), question TEXT NOT NULL,
          answer TEXT NOT NULL, sources TEXT NOT NULL, mode TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS audit (
          id INTEGER PRIMARY KEY, user_id INTEGER, action TEXT NOT NULL, detail TEXT NOT NULL,
          created TEXT DEFAULT CURRENT_TIMESTAMP);
        ''')
        for k, v in {'mode':'extractive', 'local_model':'', 'online_model':'', 'online_enabled':'false'}.items():
            db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)', (k,v))

def settings():
    with connect() as db:
        return {r['key']:r['value'] for r in db.execute('SELECT * FROM settings')}

def audit(db, user_id, action, detail=''):
    db.execute('INSERT INTO audit(user_id,action,detail) VALUES (?,?,?)', (user_id,action,detail))
