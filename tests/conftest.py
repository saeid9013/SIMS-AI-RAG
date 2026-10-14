import os
import tempfile
os.environ['SIMS_DATA_DIR']=tempfile.mkdtemp(prefix='sims-test-')
os.environ['SIMS_EMBED_MODEL_PATH']=''
os.environ['SIMS_ONLINE_API_KEY']=''
import pytest
from fastapi.testclient import TestClient
from app.main import app,attempts
from app.db import init_db,connect
from app.security import hash_password
from app.config import UPLOADS

@pytest.fixture
def client():
    init_db(); attempts.clear()
    with connect() as db:
        for table in ['history','audit','sessions','chunks_fts','chunks','documents','users']:
            db.execute(f'DELETE FROM {table}')
        db.execute("UPDATE settings SET value='extractive' WHERE key='mode'")
        db.execute("UPDATE settings SET value='false' WHERE key='online_enabled'")
        db.execute("INSERT INTO users(id,username,password,role) VALUES (1,'admin',?,'admin')",(hash_password('admin-test-password'),))
        db.execute("INSERT INTO users(id,username,password,role) VALUES (2,'reader',?,'user')",(hash_password('reader-test-password'),))
    for path in UPLOADS.iterdir():
        path.unlink()
    # Start the real lifespan; worker claims are serialized with explicit transactions.
    with TestClient(app,raise_server_exceptions=True) as test:
        # Parser checks use isolated files; integration cases can also invoke a queued job directly.
        yield test
