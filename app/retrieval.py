import array
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from functools import lru_cache
from .config import EMBED_MODEL
from . import config
from .db import connect
from .ingest import normalize

_model=None
_reranker=None
_model_lock=threading.Lock()
_rerank_lock=threading.Lock()
STOP=set('از به در و یا که را با برای این آن است چیست چه چگونه the a an is are of to in on for what how'.split())
DIGITS=str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩','01234567890123456789')

def index_text(text):
    return normalize(text).translate(DIGITS).lower()

def terms(question):
    return [w for w in re.findall(r'[^\W_]+(?:[-./][^\W_]+)*',index_text(question))
            if len(w)>1 and w not in STOP][:32]

@lru_cache(maxsize=4)
def profile():
    if not EMBED_MODEL:return ''
    path=Path(EMBED_MODEL).resolve()
    if not path.is_dir():raise ValueError('پوشه مدل embedding موجود نیست.')
    # Conservative version guard: changed path/files/mtime requires explicit reindex.
    files=[(str(p.relative_to(path)),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(path.rglob('*')) if p.is_file()]
    value=[str(path),files,config.EMBED_QUERY_PREFIX,config.EMBED_PASSAGE_PREFIX,os.getenv('SIMS_EMBED_VERSION','1')]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()

def load_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model=SentenceTransformer(EMBED_MODEL,device='cpu',local_files_only=True,trust_remote_code=False)
    return _model

def semantic_parts(parts):
    """Split at word boundaries to the actual tokenizer budget; preserve every source character."""
    if not EMBED_MODEL:return parts
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    result=[]
    with _model_lock:
        model=load_embedding_model()
        budget=int(model.max_seq_length)
        count=lambda text:len(model.tokenizer(config.EMBED_PASSAGE_PREFIX+text,add_special_tokens=True)['input_ids'])
        for location,text in parts:
            pending=text
            while count(pending)>budget:
                words=pending.split(' ')
                if len(words)<2:raise ValueError('یک واژه از بودجه tokenizer بزرگ‌تر است؛ بازبینی متن لازم است.')
                low,high=1,len(words)
                while low<high:
                    mid=(low+high+1)//2
                    if count(' '.join(words[:mid]))<=budget:low=mid
                    else:high=mid-1
                first=' '.join(words[:low])
                if count(first)>budget:raise ValueError('متن با بودجه tokenizer سازگار نیست.')
                result.append((location,first))
                pending=' '.join(words[low:])
            if pending:result.append((location,pending))
    return result

def encode(texts,query=False):
    if not EMBED_MODEL:return None
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    with _model_lock:
        model=load_embedding_model()
        prefix=config.EMBED_QUERY_PREFIX if query else config.EMBED_PASSAGE_PREFIX
        if any(len(model.tokenizer(prefix+t,add_special_tokens=True)['input_ids'])>int(model.max_seq_length) for t in texts):
            raise ValueError('متن از بودجه tokenizer embedding بیشتر است؛ پرسش را کوتاه یا اسناد را بازپردازش کنید.')
        vectors=model.encode([prefix+t for t in texts],normalize_embeddings=True,
                              batch_size=8,show_progress_bar=False)
        return [array.array('f',v).tobytes() for v in vectors]

def rerank(question,rows):
    global _reranker
    if not config.RERANK_MODEL or not rows:return rows
    if not Path(config.RERANK_MODEL).is_dir():raise ValueError('پوشه reranker محلی موجود نیست.')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    with _rerank_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker=CrossEncoder(config.RERANK_MODEL,device='cpu',local_files_only=True,trust_remote_code=False)
        scores=_reranker.predict([(question,r['text']) for r in rows],batch_size=4,show_progress_bar=False)
    if len(scores)!=len(rows):raise ValueError('تعداد نمره‌های reranker نامعتبر است.')
    ranked=sorted(zip(scores,rows),key=lambda x:float(x[0]),reverse=True)
    minimum=os.getenv('SIMS_RERANK_MIN_SCORE','').strip()
    return [r for score,r in ranked if not minimum or float(score)>=float(minimum)]

def diversify(rows,limit):
    selected=[]
    for row in rows:
        tokens=set(re.findall(r'[^\W_]+',index_text(row['text'])))
        redundant=False
        for other in selected:
            if row['document_id']!=other['document_id'] or row['location']!=other['location']:continue
            other_tokens=set(re.findall(r'[^\W_]+',index_text(other['text'])))
            union=tokens|other_tokens
            numbers=lambda t:set(re.findall(r'\d+(?:[.,]\d+)*',index_text(t)))
            if union and numbers(row['text'])==numbers(other['text']) and len(tokens&other_tokens)/len(union)>.92:
                redundant=True;break
        if not redundant:selected.append(row)
        if len(selected)>=limit:break
    return selected

def search(question,limit=5):
    tokens=terms(question);lexical=[]
    if tokens:
        expression=' OR '.join('"'+t.replace('"','""')+'"' for t in tokens)
        with connect() as db:
            lexical=[dict(r) for r in db.execute('''SELECT c.*,d.name,d.revision,bm25(chunks_fts) AS rank
              FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid JOIN documents d ON d.id=c.document_id
              WHERE chunks_fts MATCH ? AND d.status='ready' ORDER BY rank LIMIT 40''',(expression,))]
        # Prefer exact equipment tags over passages sharing generic words.
        tags=[t for t in tokens if re.search(r'[a-z].*\d|\d.*[a-z]',t)]
        if tags:
            lexical.sort(key=lambda r:sum(bool(re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)',index_text(r['text']))) for t in tags),reverse=True)
    semantic=[]
    if EMBED_MODEL:
        signature=profile()
        with connect() as db:
            stale=db.execute("""SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.document_id
                 WHERE d.status='ready' AND (c.embedding IS NULL OR c.embedding_profile!=?)""",(signature,)).fetchone()[0]
        if stale:raise ValueError('بردارهای اسناد با مدل فعلی سازگار نیستند؛ همه اسناد را بازپردازش کنید.')
        q=array.array('f');q.frombytes(encode([question],query=True)[0])
        with connect() as db:
            rows=db.execute('''SELECT c.*,d.name,d.revision FROM chunks c JOIN documents d ON d.id=c.document_id
                WHERE d.status='ready' AND c.embedding IS NOT NULL AND c.embedding_profile=?''',(signature,))
            for row in rows:
                v=array.array('f');v.frombytes(row['embedding'])
                if len(v)!=len(q):raise ValueError('ابعاد embedding ناسازگار است؛ بازپردازش لازم است.')
                score=sum(a*b for a,b in zip(q,v))
                if score>=float(os.getenv('SIMS_SEMANTIC_THRESHOLD','0.40')):semantic.append((score,dict(row)))
        semantic=[r for _,r in sorted(semantic,key=lambda x:x[0],reverse=True)[:40]]
    scores,records={},{}
    for results in (lexical,semantic):
        for rank,row in enumerate(results,1):
            cid=row['id'];records[cid]=row;scores[cid]=scores.get(cid,0)+1/(60+rank)
    candidates=[records[cid] for cid in sorted(scores,key=lambda cid:scores[cid],reverse=True)[:40]]
    selected=diversify(rerank(question,candidates),limit)
    return [dict(id=r['id'],document_id=r['document_id'],name=r['name'],revision=r['revision'],
                 location=r['location'],text=r['text'],label=f'S{i}') for i,r in enumerate(selected,1)]
