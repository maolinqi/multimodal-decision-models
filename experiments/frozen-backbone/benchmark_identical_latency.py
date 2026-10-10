"""Matched loaded-model response latency: native generation versus frozen decision.

Same checkpoint, image, question, options, precision, attention and candidate labels.
All paths use exactly the same initial information and prompt.
All repetitions and failures are retained. No training, answer-based selection or cache reuse.
"""
import argparse,gc,hashlib,json,math,os,random,re,statistics,time
from pathlib import Path
import torch
from PIL import Image
from evaluate import PROMPT,LABELS,MODELS
ROOT=Path(__file__).resolve().parent

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):q=p.with_suffix('.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2));q.replace(p)
def quantile(xs,p):return sorted(xs)[max(0,math.ceil(len(xs)*p)-1)]
def prepare(actor,row):
 path=ROOT/row['image'];b=path.read_bytes()
 if sha(b)!=row['image_sha256']:raise ValueError('Image hash mismatch')
 im=Image.open(path).convert('RGB');state=row.get('state',row.get('context',''))
 instruction=PROMPT
 prompt=instruction+'\n状态：'+json.dumps(state,ensure_ascii=False,separators=(',',':'))+'\n问题：'+row['question']+'\n选项：\n'+'\n'.join(f'{LABELS[i]}: {v}' for i,v in enumerate(row['options']))
 if actor.kind=='qwen':text='<|vision_start|><|image_pad|><|vision_end|>\n'+prompt+'\nAnswer:\n'
 else:text=actor.proc.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 inputs=actor.proc(text=[text],images=[im],return_tensors='pt').to(device='cuda',dtype=torch.bfloat16)
 if inputs['input_ids'].shape[-1]>8192:raise ValueError('Input exceeds 8192 tokens')
 return inputs,sha(text.encode())

def parse_answer(text,n):
 m=re.match(r'^\s*([A-Z])(?:\s|$|[.,:;)])',text)
 if not m:return None
 letter=m.group(1)
 index=ord(letter)-ord('A');return index if 0<=index<n else None

@torch.inference_mode()
def run_mode(actor,row,mode):
 torch.cuda.synchronize();start=time.perf_counter();inputs,prompt_sha=prepare(actor,row);torch.cuda.synchronize();prepared=time.perf_counter()
 torch.cuda.reset_peak_memory_stats();n=len(row['options']);length=int(inputs['input_ids'].shape[-1]);extra={}
 if mode=='decision':
  logits=actor.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
  torch.cuda.synchronize();compute_end=time.perf_counter();p=torch.softmax(logits[actor.labels[:n]],0).cpu().tolist();prediction=max(range(n),key=p.__getitem__);tokens=0
  extra['probabilities']=p;extra['label_mass']=float((logits[actor.labels[:n]].logsumexp(0)-logits.logsumexp(0)).exp())
 elif mode=='first_step_control':
  output=actor.model.generate(**inputs,max_new_tokens=1,do_sample=False,use_cache=True,return_dict_in_generate=True,output_logits=True)
  torch.cuda.synchronize();compute_end=time.perf_counter();logits=output.logits[0][0].float();p=torch.softmax(logits[actor.labels[:n]],0).cpu().tolist();prediction=max(range(n),key=p.__getitem__);tokens=int(output.sequences.shape[-1]-length)
  # Retention comparison on the EXACT same prepared tensors; outside the timed native call.
  direct=actor.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float();finite=torch.isfinite(logits)&torch.isfinite(direct)
  extra.update(full_vocab_max_abs_diff=float((logits[finite]-direct[finite]).abs().max()),finite_pattern_matches=bool(torch.equal(torch.isfinite(logits),torch.isfinite(direct))),candidate_max_abs_diff=float((logits[actor.labels[:n]]-direct[actor.labels[:n]]).abs().max()),candidate_agreement=int(logits[actor.labels[:n]].argmax())==int(direct[actor.labels[:n]].argmax()))
 else:
  cap=32
  output=actor.model.generate(**inputs,max_new_tokens=cap,do_sample=False,use_cache=True,return_dict_in_generate=True)
  torch.cuda.synchronize();compute_end=time.perf_counter();new_ids=output.sequences[0,length:];tokens=int(new_ids.numel());text=actor.proc.tokenizer.decode(new_ids,skip_special_tokens=True);prediction=parse_answer(text,n)
  extra.update(generated_text=text,max_new_tokens=cap,hit_token_cap=tokens==cap)
 torch.cuda.synchronize();end=time.perf_counter()
 # The extra retention forward is excluded from the control timing, including its readout.
 response_end=compute_end if mode=='first_step_control' else end
 return dict(prediction=prediction,correct=prediction==row['gold'],valid_answer=prediction is not None,generated_tokens=tokens,input_tokens=length,prompt_sha256=prompt_sha,prepare_ms=(prepared-start)*1000,compute_ms=(compute_end-prepared)*1000,response_ms=(response_end-start)*1000,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,**extra)

def bootstrap(pairs,seed=20261010):
 rng=random.Random(seed);ratios=[];deltas=[]
 for _ in range(2000):
  selected=[pairs[rng.randrange(len(pairs))] for _ in pairs];a=statistics.median(p[0] for p in selected);b=statistics.median(p[1] for p in selected);ratios.append(a/b);deltas.append(a-b)
 return dict(speedup_ratio_ci95=[quantile(ratios,.025),quantile(ratios,.975)],median_latency_difference_ci95_ms=[quantile(deltas,.025),quantile(deltas,.975)],resampling_unit='question; all repetitions first collapsed to per-question medians',bootstrap_resamples=2000)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['qwen','gemma'],required=True);ap.add_argument('--records',required=True);ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--run-name');a=ap.parse_args()
 raw=(ROOT/a.records).read_bytes();rows=list(map(json.loads,raw.decode().splitlines()));rows=rows[:a.limit] if a.limit else rows;name=a.run_name or a.model;assert all(c.isalnum() or c in '-_' for c in name);out=ROOT/'outputs/identical-latency'/name;out.mkdir(parents=True,exist_ok=True)
 script_sha=sha(Path(__file__).read_bytes());protocol=dict(model_repo=MODELS[a.model][0],model_revision=MODELS[a.model][1],dataset_sha256=sha(raw),script_sha256=script_sha,n=len(rows),repeats=a.repeats,warmup_pairs=min(5,len(rows)),limit=a.limit,seed=20261010,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],dtype='bfloat16',attention='sdpa',gemma_image_tokens=280,training=False,adapter=None,cache_reuse_between_calls=False,decision_use_cache=False,generation_use_cache=True,answer_only_max_new_tokens=32,scope='loaded-model single-request response, including image/text preparation, synchronized model compute and readout; excluding model load, HTTP/network and queue time',torch_version=torch.__version__,transformers_version=__import__('transformers').__version__)
 inherited_path=out/'inherited-protocol.json'
 if inherited_path.exists():protocol['inherited_same_input_measurements']=json.loads(inherited_path.read_text())
 if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise RuntimeError('Protocol changed; preserve old run and choose another directory')
 save(out/'protocol.json',protocol);save(out/'status.json',dict(state='loading',pid=os.getpid()))
 if a.model=='qwen':from evaluate_qwen_v2 import FrozenDecision
 else:from evaluate import FrozenDecision
 actor=FrozenDecision(a.model);versions={n:(id(p),p._version) for n,p in actor.model.named_parameters()}
 # Both paths warm up before any timed sample. This is deterministic, not correctness-filtered.
 for row in rows[:5]:run_mode(actor,row,'decision');run_mode(actor,row,'answer_only')
 plan=[];rng=random.Random(20261010);order=rows.copy();rng.shuffle(order)
 for repeat in range(a.repeats):
  ordered=order[repeat:]+order[:repeat]
  for row in ordered:
   modes=['decision','answer_only'];rng.shuffle(modes)
   for mode in modes:plan.append((row,mode,repeat))
 # Raw-logit retention control on every input, outside the main timing pairs.
 for row in order:plan.append((row,'first_step_control',0))
 progress=out/'measurements.jsonl';done={}
 if progress.exists():done={x['key']:x for x in map(json.loads,progress.read_text().splitlines())}
 with progress.open('a',buffering=1) as f:
  for row,mode,repeat in plan:
   key=f'{row["id"]}|{mode}|{repeat}'
   if key in done:continue
   try:r=run_mode(actor,row,mode)
   except Exception as e:r=dict(prediction=None,correct=False,valid_answer=False,error=type(e).__name__,error_message=str(e)[:500])
   r.update(timing_session='resumed-or-identical-run',key=key,id=row['id'],task=row['task'],gold=row['gold'],mode=mode,repeat=repeat);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());done[key]=r
   if len(done)%25==0:save(out/'status.json',dict(state='running',completed=len(done),total=len(plan)));print(a.model,'latency records',len(done),'/',len(plan),flush=True)
 after={n:(id(p),p._version) for n,p in actor.model.named_parameters()};assert versions==after,'Parameter mutation detected'
 modes={}
 for mode in sorted({x['mode'] for x in done.values()}):
  allr=[x for x in done.values() if x['mode']==mode];valid=[x for x in allr if 'error' not in x]
  ms=[x['response_ms'] for x in valid]
  modes[mode]=dict(n=len(allr),errors=len(allr)-len(valid),correct=sum(x['correct'] for x in allr),invalid_answers=sum(not x['valid_answer'] for x in allr),response_median_ms=statistics.median(ms),response_p95_ms=quantile(ms,.95),compute_median_ms=statistics.median(x['compute_ms'] for x in valid),prepare_median_ms=statistics.median(x['prepare_ms'] for x in valid),generated_tokens_median=statistics.median(x['generated_tokens'] for x in valid),hit_token_cap=sum(x.get('hit_token_cap',False) for x in valid))
 paired={}
 for baseline,decision in [('answer_only','decision')]:
  pairs=[]
  for row in rows:
   aa=[x['response_ms'] for x in done.values() if x['id']==row['id'] and x['mode']==baseline and 'error' not in x];bb=[x['response_ms'] for x in done.values() if x['id']==row['id'] and x['mode']==decision and 'error' not in x]
   if aa and bb:pairs.append((statistics.median(aa),statistics.median(bb)))
  native=statistics.median(x[0] for x in pairs);direct=statistics.median(x[1] for x in pairs)
  paired[baseline]=dict(n_questions=len(pairs),native_median_ms=native,decision_median_ms=direct,difference_ms=native-direct,speedup_ratio=native/direct,reduction_percent=(1-direct/native)*100,median_per_question_ratio=statistics.median(x[0]/x[1] for x in pairs),**bootstrap(pairs))
 control=[x for x in done.values() if x['mode']=='first_step_control' and 'error' not in x]
 retention=dict(n=len(control),candidate_agreements=sum(x['candidate_agreement'] for x in control),max_candidate_abs_diff=max(x['candidate_max_abs_diff'] for x in control),max_full_vocab_abs_diff=max(x['full_vocab_max_abs_diff'] for x in control),all_finite_patterns_match=all(x['finite_pattern_matches'] for x in control),parameter_versions_unchanged=True)
 summary=dict(protocol=protocol,modes=modes,paired=paired,retention=retention,complete=len(done)==len(plan),limitations=['Task metrics are diagnostic public/reconstructed sets, not author-pixel replication or blind non-inferiority','Every path receives the same initial information, prompt, image, state, question and options','First-step control checks candidate retention, not native free-text answer equivalence','Server contains unrelated GPU1 training; benchmark uses GPU0, host CPU/load remains shared','Generation failures/invalid/truncated answers are retained; do not claim accuracy superiority from speed alone'])
 save(out/'summary.json',summary);save(out/'status.json',dict(state='complete',records=len(done)));print(json.dumps(paired),flush=True)
if __name__=='__main__':main()
