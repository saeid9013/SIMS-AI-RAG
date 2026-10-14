import logging
import threading
from pathlib import Path
from .db import connect
from .config import UPLOADS, EMBED_MODEL
from .ingest import extract, normalize
from .retrieval import encode,profile,index_text,semantic_parts

logger = logging.getLogger(__name__)

def process_document(doc_id):
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM documents WHERE id=? AND status='queued'",(doc_id,)).fetchone()
        if not row:
            return
        db.execute("UPDATE documents SET status='processing',error='' WHERE id=?",(doc_id,))
    try:
        parts,warnings = extract(UPLOADS/row['storage'],Path(row['storage']).suffix,
                               ocr=row['warnings']=='ocr-requested')
        parts=semantic_parts(parts) if EMBED_MODEL else parts
        vectors = encode([text for _,text in parts]) if EMBED_MODEL else None
        signature=profile() if EMBED_MODEL else ''
        with connect() as db:
            # Delete/reprocess actions are blocked while processing; transaction is atomic.
            for i,(location,text) in enumerate(parts):
                cur=db.execute('INSERT INTO chunks(document_id,location,text,embedding,embedding_profile) VALUES (?,?,?,?,?)',
                               (doc_id,location,text,vectors[i] if vectors else None,signature))
                db.execute('INSERT INTO chunks_fts(rowid,text) VALUES (?,?)',(cur.lastrowid,index_text(text)))
            db.execute("UPDATE documents SET status='ready',chunk_count=?,warnings=? WHERE id=?",
                       (len(parts),'\n'.join(warnings),doc_id))
    except Exception as exc:
        logger.exception('Document %s processing failed',doc_id)
        message = str(exc) if isinstance(exc,ValueError) else 'پردازش فایل ناموفق بود؛ گزارش سرور و فرمت فایل بررسی شود.'
        with connect() as db:
            db.execute("UPDATE documents SET status='failed',error=? WHERE id=?",(message[:500],doc_id))

class Worker:
    def __init__(self):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run,daemon=True,name='document-worker')
    def start(self):
        with connect() as db:
            db.execute("UPDATE documents SET status='queued' WHERE status='processing'")
            # Never restart costly outbound teacher calls or a training run automatically.
            db.execute("UPDATE learning_jobs SET status='failed',error='سرویس هنگام اجرا متوقف شد؛ پس از بررسی کار جدید بسازید.' WHERE status='running'")
            db.execute("UPDATE training_runs SET status='failed',error='سرویس هنگام آموزش متوقف شد.' WHERE status='running'")
        self.thread.start()
    def run(self):
        while not self.stop.is_set():
            with connect() as db:
                row=db.execute("SELECT id FROM documents WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if row:
                process_document(row['id'])
            else:
                from .learning import process_next_job
                if not process_next_job(self.stop):
                    self.stop.wait(1)
    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
