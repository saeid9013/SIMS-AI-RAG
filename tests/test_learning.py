import array
import json
import re
from pathlib import Path
import pytest
from app import learning,config,retrieval
from app.db import connect,settings
from tests.test_system import login,csrf,add_document
from scripts.train_lora import tokenize_sample

def sample_rows(n=15):
    return [dict(id=i,document_sha=f'doc-{i%3}',location=f'صفحه {i}',question=f'Q{i}',answer=f'A{i}') for i in range(n)]

def seed_samples():
    docs=[add_document(f'V-{i} design pressure is 12 bar.',f'doc-{i}.txt') for i in range(3)]
    with connect() as db:
        for i in range(15):
            doc=db.execute('SELECT * FROM documents WHERE id=?',(docs[i%3],)).fetchone()
            db.execute('''INSERT INTO learning_samples(document_id,chunk_id,document_sha,name,revision,location,passage,
                question,answer,evidence,fingerprint,review) VALUES (?,1,?,?,'A',?,'Design pressure is 12 bar.',?,?,?,?,'approved')''',
                (doc['id'],doc['sha256'],doc['name'],f'صفحه {i}',f'Question {i}',f'Pressure 12 bar ({i})','Design pressure is 12 bar.',str(i)))
    return docs

def test_learning_admin_only_csrf(client):
    login(client,'reader','reader-test-password')
    assert client.get('/learning').status_code==403
    token=csrf(client.get('/'))
    assert client.post('/learning/teacher',data={'csrf':token}).status_code==403
    login(client)
    assert client.get('/learning').status_code==200
    assert client.post('/learning/train',data={'csrf':'bad'}).status_code==403

def test_teacher_grounding_rejects_invented_numbers():
    passage='The design pressure is 12 bar, temperature is 80 C.'
    good=dict(question='فشار؟',answer='۱۲ bar',evidence='The design pressure is 12 bar')
    assert learning.validate_example(good,passage)
    assert learning.validate_example({**good,'answer':'999 bar'},passage) is None
    assert learning.validate_example({**good,'evidence':'not in source'},passage) is None

def test_teacher_requires_consent_and_permission(client,monkeypatch):
    doc=add_document();monkeypatch.setattr(config,'ONLINE_KEY','test-key')
    with pytest.raises(ValueError,match='تأیید'):learning.create_teacher_job([doc],1,False)
    with pytest.raises(ValueError,match='ابتدا'):learning.create_teacher_job([doc],1,True)

def test_teacher_snapshot_dedup_review_flow(client,monkeypatch):
    doc=add_document();monkeypatch.setattr(config,'ONLINE_KEY','test-key')
    with connect() as db:
        db.execute("UPDATE settings SET value='true' WHERE key='online_enabled'")
        db.execute("UPDATE settings SET value='teacher' WHERE key='online_model'")
    job=learning.create_teacher_job([doc],1,True)
    def fake(passage,cfg):
        good=dict(question='فشار طراحی؟',answer='۱۲ bar',evidence='Design pressure of V-101 is 12 bar.')
        return [good,good,{**good,'answer':'999 bar'}]
    monkeypatch.setattr(learning,'teacher_examples',fake)
    # Directly process after worker has a chance to claim it: serialized helper is idempotent.
    learning.process_next_job()
    import time
    for _ in range(100):
        with connect() as db:
            row=db.execute('SELECT * FROM learning_jobs WHERE id=?',(job,)).fetchone()
        if row['status'] in ('completed','failed'):break
        time.sleep(.01)
    assert row['status']=='completed',dict(row)
    with connect() as db:
        samples=db.execute('SELECT * FROM learning_samples').fetchall()
    assert len(samples)==1 and samples[0]['review']=='pending'
    login(client);token=csrf(client.get('/'))
    assert client.post(f"/learning/samples/{samples[0]['id']}/review",data={'csrf':token,'decision':'approved'}).status_code==200
    with connect() as db:
        assert db.execute('SELECT review FROM learning_samples').fetchone()[0]=='approved'

def test_grouped_split_has_no_source_overlap():
    splits,unit=learning.split_samples(sample_rows())
    assert unit=='document'
    groups={k:{s['document_sha'] for s in v} for k,v in splits.items()}
    assert not groups['train']&groups['test'] and not groups['train']&groups['validation']
    rows=sample_rows()
    for row in rows:row['document_sha']='single-doc'
    splits,unit=learning.split_samples(rows)
    assert unit=='source_location' and all(splits.values())
    for row in rows:row['location']='one-page'
    with pytest.raises(ValueError,match='سه'):learning.split_samples(rows)

def test_training_requires_setup(client):
    with pytest.raises(ValueError,match='مسیر'):learning.create_training_run(1)

def test_immutable_manifest_excludes_pending_and_deleted(client,monkeypatch,tmp_path):
    docs=seed_samples()
    monkeypatch.setattr(learning,'trainer_ready',lambda:(True,'ready'))
    monkeypatch.setattr(config,'TRAIN_BASE',str(tmp_path))
    # Prevent the real background worker from invoking optional training.
    monkeypatch.setattr(learning,'process_next_job',lambda *a:False)
    run=learning.create_training_run(1)
    with connect() as db:
        raw=db.execute('SELECT manifest FROM training_runs WHERE id=?',(run,)).fetchone()[0]
        db.execute("UPDATE learning_samples SET review='rejected'")
        frozen=json.loads(raw)
        assert sum(map(len,frozen['splits'].values()))==15
        assert db.execute('SELECT manifest FROM training_runs WHERE id=?',(run,)).fetchone()[0]==raw
        db.execute("UPDATE learning_jobs SET status='completed'")
    with pytest.raises(ValueError,match='۱۰'):learning.create_training_run(1)

def test_training_accept_requires_benchmark_and_completion(client):
    login(client);token=csrf(client.get('/'))
    with connect() as db:
        run=db.execute("INSERT INTO training_runs(status,manifest) VALUES ('trained','{}')").lastrowid
    assert client.post(f'/learning/runs/{run}/accept',data={'csrf':token}).status_code==400
    assert client.post(f'/learning/runs/{run}/accept',data={'csrf':token,'benchmark_passed':'on'}).status_code==200
    assert client.post('/admin/settings',data={'csrf':token,'mode':'adapter','active_training_run':str(run)}).status_code==200
    assert settings()['mode']=='adapter' and settings()['active_training_run']==str(run)
    assert client.post('/admin/settings',data={'csrf':token,'mode':'adapter','active_training_run':'999'}).status_code==400

def test_embedding_profile_mismatch_blocks_stale_vectors(client,monkeypatch):
    add_document()
    monkeypatch.setattr(retrieval,'EMBED_MODEL','test-local')
    monkeypatch.setattr(retrieval,'profile',lambda:'new-version')
    with pytest.raises(ValueError,match='بازپردازش'):retrieval.search('V-101')

def test_reranking_changes_order_and_uses_local_only(client,monkeypatch,tmp_path):
    monkeypatch.setattr(config,'RERANK_MODEL',str(tmp_path))
    class FakeReranker:
        def predict(self,pairs,**kwargs):return [-5,9]
    monkeypatch.setattr(retrieval,'_reranker',FakeReranker())
    rows=[dict(text='unrelated',id=1),dict(text='correct evidence',id=2)]
    assert retrieval.rerank('pressure',rows)[0]['id']==2
    monkeypatch.setenv('SIMS_RERANK_MIN_SCORE','10')
    assert retrieval.rerank('pressure',rows)==[]

def test_embedding_token_budget_preserves_all_words(monkeypatch):
    class Model:
        max_seq_length=6
        def tokenizer(self,text,**kwargs):return {'input_ids':text.split()}
    monkeypatch.setattr(retrieval,'EMBED_MODEL','test-local')
    monkeypatch.setattr(retrieval,'_model',Model())
    text='one two three four five six seven eight nine ten eleven'
    parts=retrieval.semantic_parts([('page1',text)])
    assert ' '.join(t for _,t in parts)==text
    assert all(len(('passage: '+t).split())<=6 for _,t in parts)

def test_no_silent_training_truncation_and_prompt_masking():
    class Tokenizer:
        chat_template='fake'
        def apply_chat_template(self,messages,**kwargs):
            # Prefix-preserving fixture simulates a chat model's assistant boundary.
            value=' '.join(m['content'] for m in messages)
            return value+' ANSWER' if kwargs.get('add_generation_prompt') else value
        def __call__(self,text,**kwargs):return {'input_ids':list(text.encode())}
    sample=dict(name='x',revision='1',location='page',passage='value',question='q',answer='ANSWER ok')
    assert tokenize_sample(Tokenizer(),sample,max_length=5) is None

def test_assistant_only_labels_mask_prompt():
    class Tokenizer:
        chat_template='fake'
        def apply_chat_template(self,messages,**kwargs):
            if kwargs.get('add_generation_prompt'):
                return 'PROMPT|ASSISTANT:'
            return 'PROMPT|ASSISTANT:'+messages[-1]['content']
        def __call__(self,text,**kwargs):return {'input_ids':list(text.encode())}
    sample=dict(name='x',revision='1',location='page',passage='value',question='q',answer='fact')
    encoded=tokenize_sample(Tokenizer(),sample)
    prefix=len('PROMPT|ASSISTANT:'.encode())
    assert all(label==-100 for label in encoded['labels'][:prefix])
    assert encoded['labels'][prefix:]==list('fact [S1]'.encode())

def test_training_subprocess_artifacts_not_automatically_accepted(client,monkeypatch,tmp_path):
    seed_samples()
    monkeypatch.setattr(learning,'trainer_ready',lambda:(True,'ready'))
    monkeypatch.setattr(config,'TRAIN_BASE',str(tmp_path))
    monkeypatch.setattr(learning,'process_next_job',lambda *a:False)
    run=learning.create_training_run(1)
    class Process:
        returncode=0
        def __init__(self,command,**kwargs):
            directory=Path(command[-1]);(directory/'adapter').mkdir(exist_ok=True)
            (directory/'adapter'/'adapter_config.json').write_text('{}')
            (directory/'metrics.json').write_text('{"validation_loss": 1.5}')
        def poll(self):return 0
    monkeypatch.setattr(learning.subprocess,'Popen',Process)
    with connect() as db:
        job=dict(db.execute('SELECT * FROM learning_jobs').fetchone())
    learning.process_training(job)
    with connect() as db:
        row=db.execute('SELECT * FROM training_runs WHERE id=?',(run,)).fetchone()
    assert row['status']=='trained' and row['accepted_by'] is None
    assert settings()['active_training_run']==''

def test_cancel_queued_teacher_makes_no_online_calls(client,monkeypatch):
    login(client);token=csrf(client.get('/'))
    with connect() as db:
        job=db.execute("INSERT INTO learning_jobs(kind,payload) VALUES ('teacher','{}')").lastrowid
    monkeypatch.setattr(learning,'teacher_examples',lambda *a:pytest.fail('Teacher should not run'))
    assert client.post(f'/learning/jobs/{job}/cancel',data={'csrf':token}).status_code==200
    assert learning.process_next_job() is False
