"""Audit all fixed-input latency repetitions; never remove short or capped outputs."""
import argparse,hashlib,json,math,random,statistics
from pathlib import Path
from audit_natural_answers import parse
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--suite',required=True);a=p.parse_args()
root=Path(a.run);suite_bytes=Path(a.suite).read_bytes();suite=json.loads(suite_bytes);proto=json.loads((root/'protocol.json').read_text());old=json.loads((root/'summary.json').read_text())
assert old['complete'] and old['parameter_versions_unchanged']
assert hashlib.sha256(suite_bytes).hexdigest()==proto['suite_sha256']
ids=proto['ids'];assert len(ids)==100 and len(set(ids))==100 and proto['repeats']==3
source={r['id']:r for r in suite['rows']};assert set(ids)==set(source)
raw_bytes=(root/'measurements.jsonl').read_bytes();rows=[json.loads(x) for x in raw_bytes.decode().splitlines()];assert len(rows)==700
lookup={(r['id'],r['mode'],r.get('repeat')):r for r in rows};assert len(lookup)==len(rows)
pairs=[];audited=[];controls=[]
for ident in ids:
 rsource=source[ident];native=[];decision=[];hashes=set()
 ctrl=lookup[(ident,'first_step_control',None)];controls.append(ctrl)
 assert ctrl['full_vocab_max_abs_diff']==ctrl['candidate_max_abs_diff']==0 and ctrl['finite_pattern_matches']
 for repeat in range(3):
  for mode,times in [('native',native),('decision',decision)]:
   r=lookup[(ident,mode,repeat)];assert r['expected']==rsource['expected'];hashes.add(r['input_sha256']);assert math.isfinite(r['response_ms']) and r['response_ms']>0;times.append(r['response_ms'])
   if mode=='native':
    options=rsource['request']['options'];idx,rule=parse(r['text'],list(options.values()));pred=list(options)[idx] if idx is not None else None
    audited.append(dict(id=ident,repeat=repeat,prediction=pred,correct=pred==r['expected'],rule=rule,truncated=r['hit_token_cap']))
   else:assert r['generated_tokens']==0 and r['prediction'] in rsource['request']['options']
 assert len(hashes)==1
 pairs.append((statistics.median(native),statistics.median(decision)))
na=statistics.median(x[0] for x in pairs);de=statistics.median(x[1] for x in pairs)
assert math.isclose(na,old['native_median_ms'],abs_tol=1e-6) and math.isclose(de,old['decision_median_ms'],abs_tol=1e-6)
rng=random.Random(20261010);ratios=[]
for _ in range(2000):
 sample=[pairs[rng.randrange(len(pairs))] for _ in pairs];ratios.append(statistics.median(x[0] for x in sample)/statistics.median(x[1] for x in sample))
ratios.sort();nr=[r for r in rows if r['mode']=='native'];dr=[r for r in rows if r['mode']=='decision']
report=dict(complete=True,model=proto['model'],n=100,repeats=3,records=700,measurements_sha256=hashlib.sha256(raw_bytes).hexdigest(),native_median_ms=na,decision_median_ms=de,speedup=na/de,reduction_percent=(1-de/na)*100,speedup_ci95=[ratios[49],ratios[1949]],native_tokens_median=statistics.median(r['generated_tokens'] for r in nr),cap_hits=sum(r['hit_token_cap'] for r in nr),decision_correct=sum(r['prediction']==r['expected'] for r in dr),native_parsed_correct=sum(r['correct'] for r in audited),native_unresolved=sum(r['prediction'] is None for r in audited),first_step_controls=len(controls),max_full_vocab_abs_diff=0,parameter_versions_unchanged=True)
(root/'audited-native-answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audited));(root/'audited-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
