"""Versioned retrieval datasets, reviewed train-only hard negatives and embedding runs."""
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from . import config
from .db import connect,audit
from .ingest import normalize
from .learning import model_fingerprint


def init_embedding(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS embedding_pairs (
      id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL REFERENCES learning_jobs(id),
      query_id TEXT NOT NULL, query TEXT NOT NULL, positive_id TEXT NOT NULL, negative_id TEXT NOT NULL,
      positive TEXT NOT NULL, negative TEXT NOT NULL, positive_name TEXT NOT NULL, negative_name TEXT NOT NULL,
      positive_location TEXT NOT NULL, negative_location TEXT NOT NULL,
      review TEXT NOT NULL DEFAULT 'pending', reviewed_by INTEGER, reviewed_at TEXT,
      UNIQUE(job_id,query_id,negative_id));
    CREATE TABLE IF NOT EXISTS embedding_runs (
      id INTEGER PRIMARY KEY, job_id INTEGER REFERENCES learning_jobs(id), dataset_job INTEGER NOT NULL,
      status TEXT NOT NULL, manifest TEXT NOT NULL, metrics TEXT NOT NULL DEFAULT '{}',
      error TEXT NOT NULL DEFAULT '', accepted_by INTEGER, accepted_at TEXT,
      created TEXT DEFAULT CURRENT_TIMESTAMP);
    ''')
    db.execute("INSERT OR IGNORE INTO settings VALUES ('active_embedding_run','')")


def corpus_rows(db):
    return [dict(r) for r in db.execute('''SELECT c.id,c.document_id,c.location,c.text,d.sha256,d.name,d.revision
          FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.status='ready' ORDER BY c.id''')]


def signature(corpus):
    return hashlib.sha256(json.dumps(corpus,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def split_corpus(corpus):
    document_split=len({c['sha256'] for c in corpus})>=3
    unit='document' if document_split else 'source_location'
    group=lambda c:c['sha256'] if document_split else c['sha256']+'|'+c['location']
    keys=sorted({group(c) for c in corpus},key=lambda k:hashlib.sha256(k.encode()).hexdigest())
    if len(keys)<3:raise ValueError('حداقل سه سند یا محل منبع مستقل برای تفکیک داده لازم است.')
    n=max(1,len(keys)//10)
    mapping={k:('test' if i<n else 'validation' if i<2*n else 'train') for i,k in enumerate(keys)}
    return {str(c['id']):mapping[group(c)] for c in corpus},unit


def create_mining_job(user_id):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM learning_jobs WHERE status IN ('queued','running')").fetchone():
            raise ValueError('یک کار آموزشی دیگر در صف یا در حال اجراست.')
        corpus=corpus_rows(db)
        if not corpus or len(corpus)>config.EMBED_MAX_CORPUS:
            raise ValueError(f'تعداد قطعات corpus باید بین ۱ و {config.EMBED_MAX_CORPUS} باشد؛ انتخاب ناقص انجام نمی‌شود.')
        source_samples=db.execute('''SELECT s.* FROM learning_samples s JOIN documents d ON d.id=s.document_id
          AND d.sha256=s.document_sha WHERE s.review='approved' AND d.status='ready' ORDER BY s.id''').fetchall()
        assignments,unit=split_corpus(corpus)
        questions=[];seen=set()
        for s in source_samples:
            key=normalize(s['question']).lower()
            if key in seen:raise ValueError('پرسش تأییدشدهٔ تکراری وجود دارد؛ ابهام یا نسخه‌های متعارض را پیش از ساخت داده بازبینی کنید.')
            relevant=[str(c['id']) for c in corpus if c['document_id']==s['document_id']
                      and c['location']==s['location'] and normalize(s['evidence']) in normalize(c['text'])]
            if not relevant:raise ValueError('شاهد یکی از نمونه‌ها در قطعات فعلی پیدا نشد؛ نمونه را بازبینی یا دوباره تولید کنید.')
            split=assignments[relevant[0]]
            questions.append(dict(id=str(s['id']),question=s['question'],positive_ids=relevant,split=split,
                                  reviewed_by=s['reviewed_by'],reviewed_at=s['reviewed_at']))
            seen.add(key)
        if not 10<=len(questions)<=config.EMBED_MAX_QUERIES:
            raise ValueError(f'تعداد پرسش‌های تأییدشده باید بین ۱۰ و {config.EMBED_MAX_QUERIES} باشد.')
        if any(not any(q['split']==split for q in questions) for split in ('train','validation','test')):
            raise ValueError('هر سه بخش آموزش/اعتبارسنجی/آزمون به پرسش شاهددار نیاز دارند؛ از اسناد یا صفحات بیشتری نمونه بسازید.')
        payload=dict(corpus=corpus,corpus_sha256=signature(corpus),queries=questions,
                     assignments=assignments,split_unit=unit,mining='current_local_retrieval',created=time.time())
        job=db.execute("INSERT INTO learning_jobs(kind,payload,total,created_by) VALUES ('embed_mine',?,?,?)",
                       (json.dumps(payload,ensure_ascii=False),len(questions),user_id)).lastrowid
        audit(db,user_id,'embedding_dataset',str(job))
    return job


def process_mining(job,stop_event=None):
    from .retrieval import search
    data=json.loads(job['payload']);corpus={str(c['id']):c for c in data['corpus']}
    started=time.monotonic()
    for i,q in enumerate(data['queries'],1):
        if time.monotonic()-started>config.TRAIN_TIMEOUT:
            raise ValueError('استخراج منفی‌ها به سقف زمان رسید؛ کار جدید با داده کمتر بسازید.')
        with connect() as db:
            if db.execute('SELECT status FROM learning_jobs WHERE id=?',(job['id'],)).fetchone()['status']=='cancelled':return
        if stop_event and stop_event.is_set():raise ValueError('استخراج نمونه منفی به علت توقف سرویس متوقف شد.')
        if q['split']=='train':
            positive=corpus[q['positive_ids'][0]]
            candidates=search(q['question'],limit=40)
            negatives=[]
            for row in candidates:
                cid=str(row['id'])
                if cid not in corpus or cid in q['positive_ids'] or data['assignments'][cid]!='train':continue
                candidate=corpus[cid]
                # Same-source overlapped fragments are not safe negatives.
                if candidate['sha256']==positive['sha256'] and candidate['location']==positive['location']:continue
                if normalize(candidate['text'])==normalize(positive['text']):continue
                if any(normalize(corpus[p]['text']) in normalize(candidate['text']) for p in q['positive_ids']):continue
                negatives.append(candidate)
                if len(negatives)==3:break
            with connect() as db:
                for negative in negatives:
                    db.execute('''INSERT OR IGNORE INTO embedding_pairs
                        (job_id,query_id,query,positive_id,negative_id,positive,negative,positive_name,negative_name,positive_location,negative_location)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                        (job['id'],q['id'],q['question'],str(positive['id']),str(negative['id']),positive['text'],negative['text'],
                         positive['name'],negative['name'],positive['location'],negative['location']))
        with connect() as db:db.execute('UPDATE learning_jobs SET progress=? WHERE id=?',(i,job['id']))
    with connect() as db:
        db.execute("UPDATE learning_jobs SET status='completed' WHERE id=? AND status='running'",(job['id'],))


def embedding_ready():
    path=config.EMBED_TRAIN_BASE
    if not path or not Path(path).is_dir():return False,'پوشه مدل پایهٔ embedding برای آموزش تنظیم نشده است.'
    missing=[name for name in ('torch','sentence_transformers','datasets','accelerate') if importlib.util.find_spec(name) is None]
    if missing:return False,'وابستگی‌های آموزش embedding نصب نشده‌اند: '+', '.join(missing)
    return True,'پیش‌نیازهای اولیه موجودند؛ ظرفیت CPU/GPU هنگام اجرا کنترل می‌شود.'


def create_embedding_run(dataset_job,user_id):
    ok,reason=embedding_ready()
    if not ok:raise ValueError(reason)
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM learning_jobs WHERE status IN ('queued','running')").fetchone():
            raise ValueError('یک کار دیگر در صف یا در حال اجراست.')
        dataset=db.execute("SELECT * FROM learning_jobs WHERE id=? AND kind='embed_mine' AND status='completed'",(dataset_job,)).fetchone()
        if not dataset:raise ValueError('یک نسخه کامل‌شده دادهٔ بازیابی انتخاب کنید.')
        frozen=json.loads(dataset['payload'])
        if signature(corpus_rows(db))!=frozen['corpus_sha256']:
            raise ValueError('اسناد یا قطعات تغییر کرده‌اند؛ داده بازیابی را دوباره بسازید.')
        for q in frozen['queries']:
            current=db.execute('SELECT review,question FROM learning_samples WHERE id=?',(q['id'],)).fetchone()
            if not current or current['review']!='approved' or current['question']!=q['question']:
                raise ValueError('بازبینی مثبت یکی از پرسش‌ها تغییر کرده است؛ نسخه داده را دوباره بسازید.')
        approved=[dict(r) for r in db.execute("SELECT * FROM embedding_pairs WHERE job_id=? AND review='approved' ORDER BY id",(dataset_job,))]
        train_ids={q['id'] for q in frozen['queries'] if q['split']=='train'}
        if train_ids-{p['query_id'] for p in approved}:
            raise ValueError('برای همه پرسش‌های بخش آموزش حداقل یک نمونه منفی تأییدشده لازم است؛ آموزش ناقص شروع نمی‌شود.')
        if len(approved)<2:raise ValueError('حداقل دو نمونه تأییدشده برای آموزش لازم است.')
        if config.EMBED_TRAIN_LOSS not in ('triplet','mnrl'):
            raise ValueError('روش آموزش embedding باید triplet یا mnrl باشد.')
        manifest={**frozen,'loss':config.EMBED_TRAIN_LOSS,'pairs':approved,'base_model':str(Path(config.EMBED_TRAIN_BASE).resolve()),
                  'base_fingerprint':model_fingerprint(config.EMBED_TRAIN_BASE),'device':config.TRAIN_DEVICE,
                  'steps':config.TRAIN_MAX_STEPS,'query_prefix':config.EMBED_QUERY_PREFIX,
                  'passage_prefix':config.EMBED_PASSAGE_PREFIX,'created':time.time()}
        manifest['sha256']=hashlib.sha256(json.dumps(manifest,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        job=db.execute("INSERT INTO learning_jobs(kind,payload,created_by) VALUES ('embed_train','{}',?)",(user_id,)).lastrowid
        run=db.execute("INSERT INTO embedding_runs(job_id,dataset_job,status,manifest) VALUES (?,?,'queued',?)",
                       (job,dataset_job,json.dumps(manifest,ensure_ascii=False))).lastrowid
        db.execute('UPDATE learning_jobs SET payload=? WHERE id=?',(json.dumps({'run_id':run}),job))
        audit(db,user_id,'embedding_train',str(run))
    return run


def embedding_directory(run_id):return config.DATA/'embedding-training'/f'run-{int(run_id)}'


def process_embedding_training(job,stop_event=None):
    run_id=json.loads(job['payload'])['run_id']
    with connect() as db:
        row=db.execute('SELECT * FROM embedding_runs WHERE id=?',(run_id,)).fetchone()
        db.execute("UPDATE embedding_runs SET status='running' WHERE id=?",(run_id,))
    directory=embedding_directory(run_id);directory.mkdir(parents=True,exist_ok=True)
    (directory/'manifest.json').write_text(row['manifest'])
    with (directory/'trainer.log').open('wb') as output:
        proc=subprocess.Popen([sys.executable,'-m','scripts.train_embedding','--run-dir',str(directory)],stdout=output,stderr=subprocess.STDOUT)
        start=time.monotonic()
        while proc.poll() is None:
            with connect() as db:
                cancelled=db.execute('SELECT status FROM learning_jobs WHERE id=?',(job['id'],)).fetchone()['status']=='cancelled'
            if cancelled or (stop_event and stop_event.is_set()) or time.monotonic()-start>config.TRAIN_TIMEOUT:
                proc.terminate()
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise ValueError('آموزش embedding لغو شد یا به سقف زمان رسید.')
            time.sleep(.2)
    if proc.returncode or not (directory/'model'/'modules.json').is_file() or not (directory/'metrics.json').is_file():
        raise ValueError('آموزش embedding ناموفق بود؛ گزارش اجرا را بررسی کنید.')
    metrics=json.loads((directory/'metrics.json').read_text())
    validate_metrics(metrics)
    with connect() as db:
        if db.execute('SELECT status FROM learning_jobs WHERE id=?',(job['id'],)).fetchone()['status']=='cancelled':return
        db.execute("UPDATE embedding_runs SET status='trained',metrics=? WHERE id=?",(json.dumps(metrics),run_id))
        db.execute("UPDATE learning_jobs SET status='completed' WHERE id=?",(job['id'],))


def validate_metrics(metrics):
    try:
        for version in ('before','after'):
            for split in ('validation','test'):
                for metric in ('recall_at_5','mrr_at_10'):
                    value=metrics[version][split][metric]
                    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                        raise ValueError('معیارهای ارزیابی embedding معتبر نیستند.')
    except (KeyError,TypeError):
        raise ValueError('گزارش معیارهای embedding ناقص است.') from None


def accept_run(run_id,user_id):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute("SELECT * FROM embedding_runs WHERE id=? AND status='trained'",(run_id,)).fetchone()
        if not row:raise ValueError('نسخه آموزش کامل‌شده پیدا نشد.')
        metrics=json.loads(row['metrics']);validate_metrics(metrics)
        for split in ('validation','test'):
            for metric in ('recall_at_5','mrr_at_10'):
                if metrics['after'][split][metric]<metrics['before'][split][metric]:
                    raise ValueError('افت معیارهای بازیابی ثبت شده است؛ این نسخه قابل پذیرش نیست.')
        db.execute("UPDATE embedding_runs SET status='accepted',accepted_by=?,accepted_at=CURRENT_TIMESTAMP WHERE id=?",(user_id,run_id))
        audit(db,user_id,'embedding_accept',str(run_id))


def activate_run(run_id,user_id):
    from .retrieval import clear_embedding_cache
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM documents WHERE status='processing'").fetchone() or db.execute("SELECT 1 FROM learning_jobs WHERE status IN ('queued','running')").fetchone():
            raise ValueError('ابتدا پردازش اسناد و کارهای آموزشی جاری را تمام یا لغو کنید.')
        row=db.execute("SELECT * FROM embedding_runs WHERE id=? AND status='accepted'",(run_id,)).fetchone() if run_id else None
        if run_id and not row:raise ValueError('فقط نسخه پذیرفته‌شده قابل فعال‌سازی است.')
        if row:
            manifest=json.loads(row['manifest'])
            if (manifest['query_prefix'],manifest['passage_prefix'])!=(config.EMBED_QUERY_PREFIX,config.EMBED_PASSAGE_PREFIX):
                raise ValueError('پیشوندهای سرویس با مدل آموزشی یکسان نیستند؛ تنظیمات محیط را اصلاح و restart کنید.')
            if not (embedding_directory(run_id)/'model'/'modules.json').is_file():raise ValueError('فایل مدل آموزش‌دیده موجود نیست.')
        elif not config.EMBED_MODEL or not Path(config.EMBED_MODEL).is_dir():
            raise ValueError('پوشه مدل پایه برای بازگشت تنظیم نشده است.')
        # Preserve OCR needs BEFORE replacing indexed chunks.
        db.execute("""UPDATE documents SET warnings=CASE WHEN EXISTS
            (SELECT 1 FROM chunks WHERE document_id=documents.id AND location LIKE '%OCR%')
            THEN 'ocr-requested' ELSE '' END, status='queued',chunk_count=0,error=''
            WHERE status IN ('ready','queued')""")
        db.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.status='queued')")
        db.execute("DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE status='queued')")
        db.execute("UPDATE settings SET value=? WHERE key='active_embedding_run'",(str(run_id) if run_id else '',))
        audit(db,user_id,'embedding_activate',str(run_id or 'base'))
    clear_embedding_cache()
