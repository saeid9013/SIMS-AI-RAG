import hashlib
import hmac
import secrets
import time
from .db import connect

SESSION_SECONDS = 8 * 3600

def hash_password(password):
    if len(password) < 12 or len(password) > 256:
        raise ValueError('رمز باید بین ۱۲ تا ۲۵۶ نویسه باشد.')
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return f'scrypt${salt}${digest}'

def verify_password(password, encoded):
    if len(password) > 256:
        return False
    try:
        _, salt, digest = encoded.split('$')
        actual = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
        return hmac.compare_digest(actual, digest)
    except (ValueError, TypeError):
        return False

def new_session(user_id):
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with connect() as db:
        db.execute('DELETE FROM sessions WHERE expires < ?', (time.time(),))
        db.execute('INSERT INTO sessions VALUES (?,?,?,?)',
                   (hashlib.sha256(token.encode()).hexdigest(), user_id, csrf, time.time()+SESSION_SECONDS))
    return token

def session_user(token):
    if not token:
        return None
    with connect() as db:
        r = db.execute('''SELECT users.*, sessions.csrf FROM sessions JOIN users ON users.id=sessions.user_id
                       WHERE token=? AND expires>? AND users.active=1''',
                       (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
    return dict(r) if r else None

def revoke(token):
    with connect() as db:
        db.execute('DELETE FROM sessions WHERE token=?', (hashlib.sha256(token.encode()).hexdigest(),))
