import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from . import config
from .db import init_db, connect, settings, audit
from .security import session_user, verify_password, hash_password, new_session, revoke
from .retrieval import search
from .providers import answer, ProviderError
from .worker import Worker

ROOT = Path(__file__).parent
attempts = defaultdict(deque)
DUMMY_HASH = hash_password('dummy-password-for-timing')

@asynccontextmanager
async def lifespan(app):
    init_db()
    worker=Worker(); worker.start()
    yield
    worker.close()

app=FastAPI(title='SIMS-AI',version='0.2.0',lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
templates=Jinja2Templates(directory=ROOT/'templates')

@app.middleware('http')
async def security_headers(request,call_next):
    length=request.headers.get('content-length')
    if request.method=='POST':
        if length is None:
            return JSONResponse({'detail':'Content-Length is required'},status_code=411)
        try:
            if int(length)<0 or int(length)>config.MAX_UPLOAD+1024*1024:
                return JSONResponse({'detail':'حجم درخواست بیش از سقف مجاز است.'},status_code=413)
        except ValueError:
            return JSONResponse({'detail':'Invalid Content-Length'},status_code=400)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['X-Frame-Options']='DENY'
    response.headers['Referrer-Policy']='same-origin'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    response.headers['Cache-Control']='no-store'
    return response

class LoginRequired(Exception):
    pass

@app.exception_handler(LoginRequired)
async def redirect_login(request,exc):
    return RedirectResponse('/login',status_code=303)

@app.exception_handler(HTTPException)
async def http_error(request,exc):
    return templates.TemplateResponse(request=request,name='error.html',context={'user':None,
        'code':exc.status_code,'detail':exc.detail},status_code=exc.status_code)

async def user(request:Request):
    result=await run_in_threadpool(session_user,request.cookies.get('sims_session'))
    if not result:
        raise LoginRequired()
    return result

async def admin(u=Depends(user)):
    if u['role']!='admin':
        raise HTTPException(403,'این عملیات فقط برای مدیر مجاز است.')
    return u

async def form_data(request,u):
    form=await request.form()
    token=str(form.get('csrf',''))
    if not hmac.compare_digest(token,u['csrf']):
        raise HTTPException(403,'توکن امنیتی معتبر نیست؛ صفحه را دوباره باز کنید.')
    return form

def redirect(path,message=''):
    return RedirectResponse(path+('?'+'message='+quote(message) if message else ''),status_code=303)

def render(request,name,u,**context):
    return templates.TemplateResponse(request=request,name=name,context={'user':u,**context})

@app.get('/health')
async def health():
    return {'status':'ok','version':'0.2.0'}

@app.get('/login')
async def login_page(request:Request):
    token=secrets.token_urlsafe(32)
    response=render(request,'login.html',None,csrf=token,error='')
    response.set_cookie('sims_login_csrf',token,httponly=True,secure=config.COOKIE_SECURE,samesite='strict',max_age=900)
    return response

@app.post('/login')
async def login(request:Request):
    form=await request.form()
    token=str(form.get('csrf',''))
    expected=request.cookies.get('sims_login_csrf','')
    if not expected or not hmac.compare_digest(token,expected):
        raise HTTPException(403,'صفحه ورود را دوباره باز کنید.')
    username=str(form.get('username',''))[:64].strip()
    password=str(form.get('password',''))
    ip=request.client.host if request.client else 'unknown'
    now=time.monotonic()
    # Bound per-process buckets; deploy one worker, use proxy limiter for stronger protection.
    for key in list(attempts):
        while attempts[key] and now-attempts[key][0]>900:
            attempts[key].popleft()
        if not attempts[key]:
            del attempts[key]
    keys=['ip:'+ip,'user:'+username]
    if any(len(attempts[k])>=10 for k in keys):
        raise HTTPException(429,'تعداد تلاش‌ها زیاد است؛ ۱۵ دقیقه بعد تلاش کنید.')
    for k in keys:
        attempts[k].append(now)
    with connect() as db:
        row=db.execute('SELECT * FROM users WHERE username=? AND active=1',(username,)).fetchone()
    valid=await run_in_threadpool(verify_password,password,row['password'] if row else DUMMY_HASH)
    if not row or not valid:
        return render(request,'login.html',None,csrf=expected,error='نام کاربری یا رمز عبور نادرست است.')
    session=await run_in_threadpool(new_session,row['id'])
    with connect() as db:
        audit(db,row['id'],'login')
    response=redirect('/')
    response.set_cookie('sims_session',session,httponly=True,secure=config.COOKIE_SECURE,samesite='strict',max_age=28800)
    response.delete_cookie('sims_login_csrf')
    return response

@app.post('/logout')
async def logout(request:Request,u=Depends(user)):
    await form_data(request,u)
    revoke(request.cookies['sims_session'])
    response=redirect('/login'); response.delete_cookie('sims_session')
    return response

@app.get('/')
async def home(request:Request,u=Depends(user)):
    with connect() as db:
        stats={r['status']:r['n'] for r in db.execute('SELECT status,count(*) n FROM documents GROUP BY status')}
        rows=db.execute('SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT 12',(u['id'],)).fetchall()
    return render(request,'chat.html',u,cfg=settings(),stats=stats,history=rows,result=None,question='',error='')

@app.post('/ask')
async def ask(request:Request,u=Depends(user)):
    form=await form_data(request,u)
    question=str(form.get('question','')).strip()
    if not question or len(question)>2000:
        raise HTTPException(400,'پرسش باید بین ۱ تا ۲۰۰۰ نویسه باشد.')
    cfg=settings(); error=''; result=None
    try:
        sources=await run_in_threadpool(search,question)
        response=await run_in_threadpool(answer,question,sources,cfg)
        result={'answer':response,'sources':sources}
        with connect() as db:
            db.execute('INSERT INTO history(user_id,question,answer,sources,mode) VALUES (?,?,?,?,?)',
                       (u['id'],question,response,json.dumps(sources,ensure_ascii=False),cfg['mode']))
            audit(db,u['id'],'ask',cfg['mode'])
    except ProviderError as exc:
        error=str(exc)
    except Exception:
        import logging
        logging.exception('Retrieval failure')
        error='بازیابی ناموفق بود؛ تنظیمات embedding یا گزارش سرور را بررسی کنید.'
    with connect() as db:
        stats={r['status']:r['n'] for r in db.execute('SELECT status,count(*) n FROM documents GROUP BY status')}
        rows=db.execute('SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT 12',(u['id'],)).fetchall()
    return render(request,'chat.html',u,cfg=cfg,stats=stats,history=rows,result=result,question=question,error=error)

@app.get('/history/{item_id}')
async def history_item(item_id:int,request:Request,u=Depends(user)):
    with connect() as db:
        row=db.execute('SELECT * FROM history WHERE id=? AND user_id=?',(item_id,u['id'])).fetchone()
    if not row:
        raise HTTPException(404,'گفت‌وگو پیدا نشد.')
    return render(request,'history.html',u,item=row,sources=json.loads(row['sources']))

@app.get('/documents')
async def documents(request:Request,u=Depends(user)):
    with connect() as db:
        rows=db.execute('SELECT * FROM documents ORDER BY id DESC').fetchall()
    return render(request,'documents.html',u,documents=rows,max_mb=config.MAX_UPLOAD//1024//1024)

@app.post('/documents')
async def upload(request:Request,u=Depends(admin)):
    form=await form_data(request,u)
    file=form.get('file')
    if not file or not getattr(file,'filename',None):
        raise HTTPException(400,'فایل انتخاب نشده است.')
    name=Path(file.filename.replace('\\','/')).name[:200]
    suffix=Path(name).suffix.lower()
    if suffix not in ('.pdf','.docx','.doc','.txt'):
        raise HTTPException(400,'فقط PDF، DOCX، DOC و TXT مجاز است.')
    revision=str(form.get('revision','1')).strip()[:80] or '1'
    storage=secrets.token_hex(16)+suffix
    target=config.UPLOADS/storage
    digest=hashlib.sha256(); size=0
    try:
        with target.open('xb') as output:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>config.MAX_UPLOAD:
                    raise HTTPException(413,'حجم فایل بیش از سقف مجاز است.')
                digest.update(chunk); output.write(chunk)
        if size==0:
            raise HTTPException(400,'فایل خالی است.')
        with connect() as db:
            db.execute('INSERT INTO documents(name,storage,sha256,revision,created_by,warnings) VALUES (?,?,?,?,?,?)',
                       (name,storage,digest.hexdigest(),revision,u['id'],'ocr-requested' if form.get('ocr')=='on' else ''))
            audit(db,u['id'],'upload',name)
    except sqlite3.IntegrityError:
        target.unlink(missing_ok=True)
        return redirect('/documents','این فایل با همین محتوا قبلاً ثبت شده است.')
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return redirect('/documents','فایل به صف پردازش اضافه شد؛ برای وضعیت تازه صفحه را به‌روزرسانی کنید.')

@app.post('/documents/{doc_id}/delete')
async def delete_document(doc_id:int,request:Request,u=Depends(admin)):
    await form_data(request,u)
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM documents WHERE id=?',(doc_id,)).fetchone()
        if not row:
            raise HTTPException(404,'سند پیدا نشد.')
        if row['status']=='processing':
            raise HTTPException(409,'پردازش در حال انجام است؛ پس از پایان حذف کنید.')
        db.execute('DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id=?)',(doc_id,))
        db.execute('DELETE FROM documents WHERE id=?',(doc_id,))
        audit(db,u['id'],'delete_document',row['name'])
    (config.UPLOADS/row['storage']).unlink(missing_ok=True)
    return redirect('/documents','سند و قطعات آن حذف شد. متن ذخیره‌شده در تاریخچه حذف نمی‌شود.')

@app.post('/documents/{doc_id}/reprocess')
async def reprocess(doc_id:int,request:Request,u=Depends(admin)):
    form=await form_data(request,u)
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM documents WHERE id=?',(doc_id,)).fetchone()
        if not row:
            raise HTTPException(404,'سند پیدا نشد.')
        if row['status']=='processing':
            raise HTTPException(409,'سند در حال پردازش است.')
        db.execute('DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id=?)',(doc_id,))
        db.execute('DELETE FROM chunks WHERE document_id=?',(doc_id,))
        db.execute("UPDATE documents SET status='queued',chunk_count=0,error='',warnings=? WHERE id=?",
                   ('ocr-requested' if form.get('ocr')=='on' else '',doc_id))
        audit(db,u['id'],'reprocess',str(doc_id))
    return redirect('/documents','سند برای بازپردازش در صف قرار گرفت.')

@app.get('/documents/{doc_id}/download')
async def download(doc_id:int,u=Depends(user)):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM documents WHERE id=?',(doc_id,)).fetchone()
    if not row or not (config.UPLOADS/row['storage']).is_file():
        raise HTTPException(404,'فایل پیدا نشد.')
    return FileResponse(config.UPLOADS/row['storage'],filename=row['name'],media_type='application/octet-stream')

@app.get('/admin')
async def admin_page(request:Request,u=Depends(admin)):
    with connect() as db:
        users=db.execute('SELECT id,username,role,active FROM users ORDER BY id').fetchall()
        events=db.execute('SELECT * FROM audit ORDER BY id DESC LIMIT 30').fetchall()
        accepted_runs=db.execute("SELECT id FROM training_runs WHERE status='accepted' ORDER BY id DESC").fetchall()
    return render(request,'admin.html',u,cfg=settings(),users=users,events=events,
                  embedding=bool(config.EMBED_MODEL),online_key=bool(config.ONLINE_KEY),accepted_runs=accepted_runs)

@app.post('/admin/settings')
async def change_settings(request:Request,u=Depends(admin)):
    form=await form_data(request,u)
    mode=str(form.get('mode',''))
    if mode not in ('extractive','local','online','adapter'):
        raise HTTPException(400,'حالت نامعتبر است.')
    enabled='true' if form.get('online_enabled')=='on' else 'false'
    local=str(form.get('local_model','')).strip()[:120]
    online=str(form.get('online_model','')).strip()[:120]
    if mode=='online' and (enabled!='true' or not config.ONLINE_KEY or not online):
        raise HTTPException(400,'حالت آنلاین به اجازه صریح، کلید محیطی و نام مدل نیاز دارد.')
    if mode=='local' and not local:
        raise HTTPException(400,'نام مدل محلی را وارد کنید.')
    active_run=str(form.get('active_training_run','')).strip()
    if mode=='adapter':
        with connect() as db:
            accepted=db.execute("SELECT id FROM training_runs WHERE id=? AND status='accepted'",(active_run,)).fetchone()
        if not accepted:
            raise HTTPException(400,'نسخه آموزشی پذیرفته‌شده را انتخاب کنید.')
    with connect() as db:
        for k,v in {'active_training_run':active_run,'mode':mode,'online_enabled':enabled,'local_model':local,'online_model':online}.items():
            db.execute('UPDATE settings SET value=? WHERE key=?',(v,k))
        audit(db,u['id'],'settings',mode)
    return redirect('/admin','تنظیمات ذخیره شد.')

@app.post('/admin/users')
async def add_user(request:Request,u=Depends(admin)):
    form=await form_data(request,u)
    username=str(form.get('username','')).strip()
    role=str(form.get('role','user'))
    if not re.fullmatch(r'[A-Za-z0-9_.-]{3,64}',username) or role not in ('admin','user'):
        raise HTTPException(400,'نام کاربری یا نقش معتبر نیست.')
    try:
        encoded=await run_in_threadpool(hash_password,str(form.get('password','')))
        with connect() as db:
            db.execute('INSERT INTO users(username,password,role) VALUES (?,?,?)',(username,encoded,role))
            audit(db,u['id'],'create_user',username)
    except ValueError as exc:
        raise HTTPException(400,str(exc))
    except sqlite3.IntegrityError:
        raise HTTPException(409,'نام کاربری تکراری است.')
    return redirect('/admin','کاربر ایجاد شد.')

@app.post('/admin/users/{user_id}/toggle')
async def toggle_user(user_id:int,request:Request,u=Depends(admin)):
    await form_data(request,u)
    if user_id==u['id']:
        raise HTTPException(400,'امکان غیرفعال کردن حساب خودتان وجود ندارد.')
    with connect() as db:
        row=db.execute('SELECT * FROM users WHERE id=?',(user_id,)).fetchone()
        if not row:
            raise HTTPException(404,'کاربر پیدا نشد.')
        db.execute('UPDATE users SET active=1-active WHERE id=?',(user_id,))
        db.execute('DELETE FROM sessions WHERE user_id=?',(user_id,))
        audit(db,u['id'],'toggle_user',row['username'])
    return redirect('/admin','وضعیت کاربر تغییر کرد.')

@app.get('/account')
async def account(request:Request,u=Depends(user)):
    return render(request,'account.html',u)

@app.post('/account/password')
async def change_password(request:Request,u=Depends(user)):
    form=await form_data(request,u)
    if not await run_in_threadpool(verify_password,str(form.get('current','')),u['password']):
        raise HTTPException(400,'رمز فعلی نادرست است.')
    if form.get('password')!=form.get('repeat'):
        raise HTTPException(400,'تکرار رمز مطابقت ندارد.')
    try:
        encoded=await run_in_threadpool(hash_password,str(form.get('password','')))
    except ValueError as exc:
        raise HTTPException(400,str(exc))
    with connect() as db:
        db.execute('UPDATE users SET password=? WHERE id=?',(encoded,u['id']))
        db.execute('DELETE FROM sessions WHERE user_id=?',(u['id'],))
        audit(db,u['id'],'change_password')
    return redirect('/login')

@app.get('/learning')
async def learning_page(request:Request,u=Depends(admin)):
    from .learning import trainer_ready
    review=request.query_params.get('review','pending')
    if review not in ('pending','approved','rejected'):
        review='pending'
    try:page=max(1,int(request.query_params.get('page','1')))
    except ValueError:page=1
    with connect() as db:
        docs=db.execute("SELECT id,name,revision,chunk_count FROM documents WHERE status='ready' ORDER BY id").fetchall()
        jobs=db.execute('SELECT id,kind,status,progress,total,error,created FROM learning_jobs ORDER BY id DESC LIMIT 20').fetchall()
        samples=db.execute('SELECT * FROM learning_samples WHERE review=? ORDER BY id DESC LIMIT 20 OFFSET ?',
                           (review,(page-1)*20)).fetchall()
        counts={r['review']:r['n'] for r in db.execute('SELECT review,COUNT(*) n FROM learning_samples GROUP BY review')}
        runs=db.execute('SELECT * FROM training_runs ORDER BY id DESC LIMIT 20').fetchall()
    ready,reason=trainer_ready()
    return render(request,'learning.html',u,documents=docs,jobs=jobs,samples=samples,counts=counts,
                  runs=runs,review=review,page=page,trainer_ready=ready,trainer_reason=reason,cfg=settings())

@app.post('/learning/teacher')
async def teacher_job(request:Request,u=Depends(admin)):
    from .learning import create_teacher_job
    form=await form_data(request,u)
    try:
        if form.get('all_documents')=='on':
            with connect() as db:
                ids=[r['id'] for r in db.execute("SELECT id FROM documents WHERE status='ready'")]
        else:ids=list({int(v) for v in form.getlist('document_id')})
        job=create_teacher_job(ids,u['id'],form.get('consent')=='on')
    except ValueError as exc:
        raise HTTPException(400,str(exc))
    return redirect('/learning',f'کار {job} ثبت شد. هر قطعه یک درخواست به مدل آنلاین خواهد داشت؛ هزینه با سرویس تنظیم‌شده است.')

@app.post('/learning/samples/{sample_id}/review')
async def review_sample(sample_id:int,request:Request,u=Depends(admin)):
    form=await form_data(request,u)
    decision=str(form.get('decision',''))
    if decision not in ('approved','rejected'):
        raise HTTPException(400,'تصمیم نامعتبر است.')
    with connect() as db:
        if not db.execute('SELECT id FROM learning_samples WHERE id=?',(sample_id,)).fetchone():
            raise HTTPException(404,'نمونه پیدا نشد.')
        db.execute('UPDATE learning_samples SET review=?,reviewed_by=?,reviewed_at=CURRENT_TIMESTAMP WHERE id=?',
                   (decision,u['id'],sample_id))
        audit(db,u['id'],'sample_review',f'{sample_id}:{decision}')
    return redirect('/learning','تصمیم بازبینی ثبت شد. نسخه‌های آموزشی قبلی تغییر نمی‌کنند.')

@app.post('/learning/train')
async def queue_training(request:Request,u=Depends(admin)):
    from .learning import create_training_run
    form=await form_data(request,u)
    if form.get('confirm_training')!='on':
        raise HTTPException(400,'شروع آموزش روی سخت‌افزار تنظیم‌شده را تأیید کنید.')
    try:run=create_training_run(u['id'])
    except ValueError as exc:raise HTTPException(400,str(exc))
    return redirect('/learning',f'نسخه آموزشی {run} از تمام نمونه‌های تأییدشده و اسناد موجود ساخته و در صف قرار گرفت.')

@app.post('/learning/jobs/{job_id}/cancel')
async def cancel_learning(job_id:int,request:Request,u=Depends(admin)):
    await form_data(request,u)
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM learning_jobs WHERE id=?',(job_id,)).fetchone()
        if not row or row['status'] not in ('queued','running'):
            raise HTTPException(409,'کار قابل لغو نیست.')
        db.execute("UPDATE learning_jobs SET status='cancelled' WHERE id=?",(job_id,))
        db.execute("UPDATE training_runs SET status='cancelled' WHERE job_id=?",(job_id,))
        audit(db,u['id'],'cancel_learning',str(job_id))
    return redirect('/learning','درخواست لغو ثبت شد. درخواست آنلاین در حال ارسال ممکن است تکمیل شود؛ درخواست جدیدی ارسال نمی‌شود.')

@app.post('/learning/runs/{run_id}/accept')
async def accept_training(run_id:int,request:Request,u=Depends(admin)):
    form=await form_data(request,u)
    if form.get('benchmark_passed')!='on':
        raise HTTPException(400,'ارزیابی مستقل کیفیت و عدم افت نسبت به مدل پایه باید تأیید شود.')
    with connect() as db:
        cur=db.execute("UPDATE training_runs SET status='accepted',accepted_by=?,accepted_at=CURRENT_TIMESTAMP WHERE id=? AND status='trained'",
                       (u['id'],run_id))
        if not cur.rowcount:raise HTTPException(409,'فقط آموزش کامل‌شده قابل پذیرش است.')
        audit(db,u['id'],'accept_training',str(run_id))
    return redirect('/learning','پذیرش ثبت شد؛ پس از اجرای سرویس adapter، آن را از تنظیم مدل انتخاب کنید. فعال‌سازی خودکار نیست.')

@app.get('/learning/runs/{run_id}/{artifact}')
async def training_artifact(run_id:int,artifact:str,u=Depends(admin)):
    from .learning import run_directory
    if artifact not in ('manifest.json','metrics.json','trainer.log'):
        raise HTTPException(404,'خروجی نامعتبر است.')
    with connect() as db:
        run=db.execute('SELECT * FROM training_runs WHERE id=?',(run_id,)).fetchone()
    if not run:raise HTTPException(404,'نسخه پیدا نشد.')
    if artifact=='manifest.json':
        from fastapi.responses import Response
        return Response(run['manifest'],media_type='application/json',
                        headers={'Content-Disposition':f'attachment; filename="run-{run_id}-manifest.json"'})
    path=run_directory(run_id)/artifact
    if not path.is_file():raise HTTPException(404,'خروجی هنوز ساخته نشده است.')
    return FileResponse(path,filename=f'run-{run_id}-{artifact}',media_type='text/plain' if artifact.endswith('.log') else 'application/json')
