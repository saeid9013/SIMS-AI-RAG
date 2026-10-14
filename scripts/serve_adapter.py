"""Serve an explicitly accepted local LoRA adapter for the main application's local-only provider."""
import argparse
import json
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',type=int,required=True)
    p.add_argument('--port',type=int,default=8001);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    from app.db import init_db,connect
    from app.learning import run_directory,model_fingerprint
    init_db()
    with connect() as db:
        run=db.execute('SELECT * FROM training_runs WHERE id=?',(args.run_id,)).fetchone()
    if not run or run['status']!='accepted':raise ValueError('Run must be trained and explicitly accepted by administrator')
    manifest=json.loads(run['manifest'])
    if model_fingerprint(manifest['base_model'])!=manifest['base_fingerprint']:
        raise ValueError('فایل‌های مدل پایه با نسخه آموزشی یکسان نیستند.')
    import threading
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel,Field
    import uvicorn
    torch.set_num_threads(5)
    tokenizer=AutoTokenizer.from_pretrained(run_directory(args.run_id)/'adapter',local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(manifest['base_model'],local_files_only=True,trust_remote_code=False,
                                             torch_dtype=torch.float32 if args.device=='cpu' else torch.float16)
    model=PeftModel.from_pretrained(model,run_directory(args.run_id)/'adapter',local_files_only=True).to(args.device)
    model.eval();lock=threading.Lock()
    service=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    class Chat(BaseModel):
        messages:list[dict[str,str]]=Field(max_length=10)
    @service.post('/api/chat')
    def chat(payload:Chat):
        if not lock.acquire(blocking=False):raise HTTPException(429,'Adapter busy')
        try:
            text=tokenizer.apply_chat_template(payload.messages,tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(text,return_tensors='pt').to(args.device)
            if inputs['input_ids'].shape[1]>7000:raise HTTPException(400,'Context too long')
            with torch.inference_mode():
                ids=model.generate(**inputs,max_new_tokens=900,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            result=tokenizer.decode(ids[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            return {'run_id':args.run_id,'message':{'content':result}}
        finally:lock.release()
    uvicorn.run(service,host='127.0.0.1',port=args.port,workers=1)

if __name__=='__main__':main()
