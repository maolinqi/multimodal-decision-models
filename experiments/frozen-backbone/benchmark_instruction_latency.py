"""Exploratory fixed-input natural generation screening, no minimum output length."""
import argparse,hashlib,json,random,statistics,time,types
from pathlib import Path
import torch
from multimodal_decision.registry import create_model
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--suite',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=100);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.45)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
def save(name,value):
 t=out/(name+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(out/name)
suite_bytes=Path(a.suite).read_bytes();suite=json.loads(suite_bytes);rows=suite['rows'][:a.limit]
save('status.json',dict(state='loading',model=a.model))
actor=create_model(a.model);assert actor.adapter is None
original=actor.context
# Both inference paths receive the same neutral task wording, without a letter-only instruction.
def context(self,request):
 content,images,provenance,keys=original(request)
 for item in content:
  if item['type']=='text':item['text']=item['text'].replace('结合实际图像与测量状态回答，只输出最合适选项的大写字母。','结合实际图像与测量状态回答问题，选择最合适的选项。')
 return content,images,provenance,keys
actor.context=types.MethodType(context,actor)
versions={k:(id(v),v._version) for k,v in actor.model.named_parameters()}
save('protocol.json',dict(model=a.model,suite_sha256=hashlib.sha256(suite_bytes).hexdigest(),ids=[r['id'] for r in rows],selection='all fixed 100 ScienceQA image-test questions, seed 42',repeats=a.repeats,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),shared_prompt='neutral choice task, no brevity/explanation/minimum length instruction',max_new_tokens=2048,training=False,torch=torch.__version__,transformers=__import__('transformers').__version__,gpu_uuid=__import__('os').environ['CUDA_VISIBLE_DEVICES'],scope='loaded-model response includes preparation and synchronized compute; GPU shared with existing training'))
rng=random.Random(42);results=[]
with torch.inference_mode():
 # Warm up vision and decision without demanding a long generated answer.
 for r in rows[:2]:
  inputs,_,_=actor.prepare(r['request']);actor.logits(inputs);torch.cuda.synchronize()
 for row in rows:
  for repeat in range(a.repeats):
   pair={};modes=['native','decision'];rng.shuffle(modes)
   for mode in modes:
    torch.cuda.synchronize();start=time.perf_counter();inputs,keys,_=actor.prepare(row['request']);input_sha=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
    if mode=='decision':
     logits=actor.logits(inputs);prediction=keys[int(logits[actor.label_ids[:len(keys)]].argmax())];tokens=0;text=None
    else:
     kw=dict(max_new_tokens=2048,do_sample=False,return_dict_in_generate=True)
     if getattr(actor,'family',None)!='internvl':kw['use_cache']=True
     output=actor.model.generate(**inputs,**kw)
     ids=output.sequences[0];length=inputs['input_ids'].shape[-1]
     # InternVL generates from inputs_embeds: HF output can contain only new IDs.
     if getattr(actor,'family',None)!='internvl':ids=ids[length:]
     tok=getattr(actor,'tokenizer',None) or actor.processor.tokenizer
     text=tok.decode(ids,skip_special_tokens=True);tokens=int(ids.numel());prediction=None
    torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
    pair[mode]=dict(id=row['id'],repeat=repeat,mode=mode,input_sha256=input_sha,response_ms=ms,generated_tokens=tokens,text=text,prediction=prediction,expected=row['expected'],hit_token_cap=tokens>=2048)
    with (out/'measurements.jsonl').open('a') as f:f.write(json.dumps(pair[mode],ensure_ascii=False)+'\n')
   assert pair['native']['input_sha256']==pair['decision']['input_sha256']
   results.append(pair);save('status.json',dict(state='running',completed=len(results),total=len(rows)*a.repeats))
  inputs,keys,_=actor.prepare(row['request'])
  direct=actor.logits(inputs)
  ctrl=actor.model.generate(**inputs,max_new_tokens=1,do_sample=False,return_dict_in_generate=True,output_logits=True).logits[0][0].float()
  finite=torch.isfinite(direct)&torch.isfinite(ctrl)
  record=dict(id=row['id'],mode='first_step_control',full_vocab_max_abs_diff=float((direct[finite]-ctrl[finite]).abs().max()),candidate_max_abs_diff=float((direct[actor.label_ids[:len(keys)]]-ctrl[actor.label_ids[:len(keys)]]).abs().max()),finite_pattern_matches=bool(torch.equal(torch.isfinite(direct),torch.isfinite(ctrl))))
  with (out/'measurements.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  assert record['full_vocab_max_abs_diff']==0 and record['candidate_max_abs_diff']==0 and record['finite_pattern_matches']
 unchanged=versions=={k:(id(v),v._version) for k,v in actor.model.named_parameters()};assert unchanged
summary=dict(complete=True,n=len(rows),repeats=a.repeats,native_median_ms=statistics.median(statistics.median(r['native']['response_ms'] for r in results if r['native']['id']==row['id']) for row in rows),decision_median_ms=statistics.median(statistics.median(r['decision']['response_ms'] for r in results if r['decision']['id']==row['id']) for row in rows),native_tokens_median=statistics.median(r['native']['generated_tokens'] for r in results),cap_hits=sum(r['native']['hit_token_cap'] for r in results),parameter_versions_unchanged=unchanged)
summary['speedup']=summary['native_median_ms']/summary['decision_median_ms'];save('summary.json',summary);save('status.json',dict(state='complete',completed=len(results),total=len(rows)*a.repeats));print(json.dumps(summary),flush=True)
