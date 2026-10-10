"""Check candidate tokens in their actual text continuation context; no model/labels used."""
import json,hashlib
from pathlib import Path
from download_models import MODELS
from transformers import AutoProcessor
from evaluate import PROMPT,LABELS
ROOT=Path(__file__).resolve().parent
t=AutoProcessor.from_pretrained(MODELS['qwen'][2],local_files_only=True).tokenizer
rows=list(map(json.loads,(ROOT/'data/qwen/records.jsonl').read_text().splitlines()));results={}
for tail in ('\nAnswer:','\nAnswer:\n'):
 checked=0;failures=[]
 for r in rows:
  prompt=PROMPT+'\n状态：'+json.dumps(r.get('state',r.get('context','')),ensure_ascii=False,separators=(',',':'))+'\n问题：'+r['question']+'\n选项：\n'+'\n'.join(f'{LABELS[i]}: {v}' for i,v in enumerate(r['options']))
  text='<|vision_start|><|image_pad|><|vision_end|>\n'+prompt+tail;pre=t.encode(text,add_special_tokens=False)
  for label in LABELS[:len(r['options'])]:
   checked+=1;ids=t.encode(label,add_special_tokens=False);joint=t.encode(text+label,add_special_tokens=False)
   if len(ids)!=1 or joint!=pre+ids:failures.append(dict(id=r['id'],label=label))
 results[repr(tail)]=dict(labels_checked=checked,mapping_failures=len(failures),failure_examples=failures[:10])
assert results[repr('\nAnswer:\n')]['mapping_failures']==0
out=dict(rows=len(rows),results=results,source_records_sha256=hashlib.sha256((ROOT/'data/qwen/records.jsonl').read_bytes()).hexdigest(),meaning='v1 tail permits colon+letter BPE merges; newline tail keeps the fixed prefix stable for every candidate. No weight/training change.')
(ROOT/'run/qwen_token_boundary_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out))
