import array
import os
import re
import threading
from .config import EMBED_MODEL
from .db import connect
from .ingest import normalize

_model = None
_model_lock = threading.Lock()
STOP = set('از به در و یا که را با برای این آن است چیست چه چگونه the a an is are of to in on for what how'.split())

def terms(question):
    return [w for w in re.findall(r'[^\W_]+(?:[-./][^\W_]+)*',normalize(question).lower())
            if len(w)>1 and w not in STOP][:32]

def encode(texts, query=False):
    global _model
    if not EMBED_MODEL:
        return None
    # Offline environment flags are set before importing any model framework.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBED_MODEL, device='cpu', local_files_only=True)
        prefix = 'query: ' if query else 'passage: '
        vectors = _model.encode([prefix+t for t in texts], normalize_embeddings=True,
                                batch_size=8, show_progress_bar=False)
        return [array.array('f',v).tobytes() for v in vectors]

def search(question, limit=5):
    tokens = terms(question)
    lexical = []
    if tokens:
        # Each literal is quoted: no user-controlled FTS operators or SQL.
        expression = ' OR '.join('"'+t.replace('"','""')+'"' for t in tokens)
        with connect() as db:
            lexical = [dict(r) for r in db.execute('''
              SELECT c.*, d.name,d.revision, bm25(chunks_fts) AS rank FROM chunks_fts
              JOIN chunks c ON c.id=chunks_fts.rowid JOIN documents d ON d.id=c.document_id
              WHERE chunks_fts MATCH ? AND d.status='ready' ORDER BY rank LIMIT 20''',(expression,))]
    semantic = []
    if EMBED_MODEL:
        query_bytes = encode([question],query=True)[0]
        q = array.array('f'); q.frombytes(query_bytes)
        with connect() as db:
            rows = db.execute('''SELECT c.*,d.name,d.revision FROM chunks c JOIN documents d
               ON d.id=c.document_id WHERE d.status='ready' AND c.embedding IS NOT NULL''')
            for r in rows:
                v = array.array('f'); v.frombytes(r['embedding'])
                if len(v) != len(q):
                    continue
                similarity = sum(a*b for a,b in zip(q,v))
                if similarity >= float(os.getenv('SIMS_SEMANTIC_THRESHOLD','0.40')):
                    semantic.append((similarity,dict(r)))
        semantic = [r for _,r in sorted(semantic,key=lambda x:x[0],reverse=True)[:20]]
    scores, records = {}, {}
    for results in (lexical, semantic):
        for rank,row in enumerate(results,1):
            cid = row['id']; records[cid] = row
            scores[cid] = scores.get(cid,0)+1/(60+rank)
    selected = sorted(scores,key=lambda cid:scores[cid],reverse=True)[:limit]
    return [dict(id=records[cid]['id'],document_id=records[cid]['document_id'],
                 name=records[cid]['name'],revision=records[cid]['revision'],
                 location=records[cid]['location'],text=records[cid]['text'],
                 label=f'S{i}') for i,cid in enumerate(selected,1)]
