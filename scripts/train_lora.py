"""Offline LoRA SFT subprocess. No network, shell commands or user-controlled code loading."""
import argparse
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['WANDB_DISABLED']='true'

def training_messages(sample):
    from app.providers import SYSTEM
    return [{'role':'system','content':SYSTEM},
            {'role':'user','content':f"شواهد:\n[S1] {sample['name']} / {sample['revision']} / {sample['location']}\n{sample['passage']}\n\nپرسش کاربر:\n{sample['question']}"},
            {'role':'assistant','content':sample['answer']+' [S1]'}]

def tokenize_sample(tokenizer,sample,max_length=2048):
    messages=training_messages(sample)
    if not tokenizer.chat_template:
        raise ValueError('مدل پایه باید chat template معتبر داشته باشد.')
    prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    prompt_ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
    ids=tokenizer(full,add_special_tokens=False)['input_ids']
    if len(ids)>max_length:
        return None  # Never truncate away evidence or silently train a partial answer.
    if ids[:len(prompt_ids)]!=prompt_ids or len(ids)<=len(prompt_ids):
        raise ValueError('قالب چت با ماسک‌کردن prompt سازگار نیست؛ قالب/مدل دیگری انتخاب کنید.')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),
            'labels':[-100]*len(prompt_ids)+ids[len(prompt_ids):]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
    directory=Path(args.run_dir).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    from app.learning import model_fingerprint
    if model_fingerprint(manifest['base_model'])!=manifest['base_fingerprint']:
        raise ValueError('نسخه فایل‌های مدل پایه از زمان ثبت کار تغییر کرده است.')
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments
    from peft import LoraConfig,get_peft_model,TaskType
    from app.config import TRAIN_CPU_MAX_PARAMS
    device=manifest['device']
    if device not in ('cpu','cuda'):
        raise ValueError('Device must be cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('GPU موجود نیست. برای مدل کوچک فقط با تصمیم مسئول سرور SIMS_TRAIN_DEVICE=cpu تنظیم شود.')
    torch.set_num_threads(5)
    tokenizer=AutoTokenizer.from_pretrained(manifest['base_model'],local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:raise ValueError('Tokenizer needs EOS or PAD')
        tokenizer.pad_token=tokenizer.eos_token
    # Inspect parameter count on a meta device BEFORE allocating weights in CPU memory.
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    cfg=AutoConfig.from_pretrained(manifest['base_model'],local_files_only=True,trust_remote_code=False)
    with init_empty_weights():
        empty=AutoModelForCausalLM.from_config(cfg,trust_remote_code=False)
    parameter_count=sum(p.numel() for p in empty.parameters());del empty
    if device=='cpu' and parameter_count>TRAIN_CPU_MAX_PARAMS:
        raise ValueError(f'مدل {parameter_count:,} پارامتری از سقف آموزش CPU بیشتر است؛ از GPU جداگانه استفاده کنید.')
    dtype=torch.float32 if device=='cpu' else torch.float16
    model=AutoModelForCausalLM.from_pretrained(manifest['base_model'],local_files_only=True,
                                              trust_remote_code=False,torch_dtype=dtype)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,
                                          lora_dropout=0.05,target_modules='all-linear',bias='none'))
    model.enable_input_require_grads()
    class Samples(Dataset):
        def __init__(self,items):
            self.items=[v for s in items if (v:=tokenize_sample(tokenizer,s)) is not None]
        def __len__(self):return len(self.items)
        def __getitem__(self,index):return self.items[index]
    datasets={key:Samples(value) for key,value in manifest['splits'].items()}
    if any(len(datasets[k])!=len(manifest['splits'][k]) for k in datasets):
        raise ValueError('حداقل یک نمونه از بودجه توکن بیشتر است؛ هیچ آموزش ناقصی انجام نشد. اندازه قطعات/مدل را اصلاح کنید.')
    def collate(rows):
        width=max(len(r['input_ids']) for r in rows)
        output={k:[] for k in ('input_ids','attention_mask','labels')}
        for row in rows:
            for key in output:
                pad=-100 if key=='labels' else tokenizer.pad_token_id if key=='input_ids' else 0
                output[key].append(row[key]+[pad]*(width-len(row[key])))
        return {k:torch.tensor(v,dtype=torch.long) for k,v in output.items()}
    arguments=TrainingArguments(output_dir=str(directory/'checkpoints'),max_steps=manifest['steps'],
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,
        learning_rate=1e-4,warmup_ratio=0.05,logging_steps=1,save_strategy='no',report_to=[],
        use_cpu=device=='cpu',fp16=device=='cuda',gradient_checkpointing=True,dataloader_num_workers=0,
        seed=42,data_seed=42)
    trainer=Trainer(model=model,args=arguments,train_dataset=datasets['train'],
                    eval_dataset=datasets['validation'],data_collator=collate)
    baseline=trainer.evaluate(metric_key_prefix='base_validation')
    result=trainer.train()
    validation=trainer.evaluate(metric_key_prefix='validation')
    # Test is observed only after training; it is not used to choose optimization steps.
    heldout=trainer.evaluate(eval_dataset=datasets['test'],metric_key_prefix='test')
    model.save_pretrained(directory/'adapter',safe_serialization=True)
    tokenizer.save_pretrained(directory/'adapter')
    metrics={**baseline,**result.metrics,**validation,**heldout,'parameters':parameter_count,
             'split_counts':{k:len(v) for k,v in datasets.items()},'dataset_sha256':manifest['sha256'],
             'quality_approved':False,'note':'Loss is not answer correctness. Human held-out QA evaluation required.'}
    (directory/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print('LoRA adapter saved; NOT activated. Human quality review is required.',flush=True)

if __name__=='__main__':main()
