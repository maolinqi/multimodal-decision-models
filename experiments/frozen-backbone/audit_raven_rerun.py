"""Fixed 300 RAVEN questions: explicit input ablations, frozen weights, all results kept."""
import hashlib,json,os,time
from pathlib import Path
import torch
from PIL import Image
from evaluate_qwen_v2 import FrozenDecision,PROMPT
ROOT=Path(__file__).resolve().parent
out=ROOT/'outputs/raven-input-audit-v3';out.mkdir(parents=True,exist_ok=True)
def save(name,x):
 p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
raw=(ROOT/'data/qwen/records.jsonl').read_bytes();rows=[json.loads(x) for x in raw.decode().splitlines() if json.loads(x)['task']=='raven'];assert len(rows)==300
save('protocol.json',dict(n=300,records_sha256=hashlib.sha256(raw).hexdigest(),variants=['original_zh','english_newline','author_slot','english_enlarged'],training=False,scope='same fixed questions/options/golds; image enlargement adds no source detail; diagnostic input ablations, not sealed test selection'))
actor=FrozenDecision('qwen');tok=actor.proc.tokenizer;versions={k:(id(v),v._version) for k,v in actor.model.named_parameters()}
results=[];image_audit=[]
with torch.inference_mode(),(out/'predictions.jsonl').open('w') as f:
 for idx,row in enumerate(rows):
  b=(ROOT/row['image']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['image_sha256'];im=Image.open(ROOT/row['image']).convert('RGB');assert row['options']==['figure '+x for x in 'ABCDEFGH'] and 0<=row['gold']<8
  image_audit.append(dict(id=row['id'],size=list(im.size),image_sha256=row['image_sha256'],gold=row['gold']))
  for variant in ['original_zh','english_newline','author_slot','english_enlarged']:
   if variant=='original_zh':
    text=PROMPT+'\n状态：'+json.dumps(row['context'],ensure_ascii=False,separators=(',',':'))+'\n问题：'+row['question']+'\n选项：\n'+'\n'.join(f'{chr(65+i)}: {v}' for i,v in enumerate(row['options']))+'\nAnswer:\n'
   else:
    text='Context:\n'+row['context']+'\n\nQuestion: '+row['question']+'\nOptions:'+''.join(f'\n({chr(65+i)}) {v}' for i,v in enumerate(row['options']))+ ('\nAnswer: (' if variant=='author_slot' else '\nAnswer:\n')
   text='<|vision_start|><|image_pad|><|vision_end|>'+('' if variant=='author_slot' else '\n')+text
   prefix=tok.encode(text,add_special_tokens=False)
   assert all(tok.encode(text+c,add_special_tokens=False)==prefix+[actor.labels[i]] for i,c in enumerate('ABCDEFGH')), 'unstable label token boundary'
   image=im.resize((im.width*2,im.height*2),Image.Resampling.LANCZOS) if variant=='english_enlarged' else im
   inputs=actor.proc(text=[text],images=[image],return_tensors='pt').to(device='cuda',dtype=torch.bfloat16)
   logits=actor.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float();sel=logits[actor.labels[:8]];pred=int(sel.argmax());record=dict(id=row['id'],variant=variant,gold=row['gold'],prediction=pred,correct=pred==row['gold'],probabilities=torch.softmax(sel,0).cpu().tolist(),input_tokens=int(inputs['input_ids'].shape[-1]),image_grid_thw=inputs['image_grid_thw'].cpu().tolist(),prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
   f.write(json.dumps(record)+'\n');f.flush();results.append(record)
  save('status.json',dict(state='running',completed=idx+1,total=300))
 unchanged=versions=={k:(id(v),v._version) for k,v in actor.model.named_parameters()};assert unchanged
save('image-audit.json',image_audit)
save('summary.json',dict(complete=True,parameter_versions_unchanged=unchanged,variants={v:dict(n=300,correct=sum(r['correct'] for r in results if r['variant']==v),accuracy=sum(r['correct'] for r in results if r['variant']==v)/300) for v in {r['variant'] for r in results}}))
save('status.json',dict(state='complete',completed=300,total=300))
