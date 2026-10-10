"""Reparse stored natural outputs without rerunning inference or changing timings.
The original timing parser misses labels such as '选项是：**B**'. This audit
handles formatting only; it never uses gold when selecting a label.
"""
import argparse,hashlib,json,re
from pathlib import Path

def parse(text,options):
 n=len(options)
 m=re.match(r'^\s*([A-Z])(?=\s|$|[.,:;)、，。])',text)
 if m and ord(m[1])-65<n:return ord(m[1])-65,'leading_label'
 matches=re.findall(r'(?:答案|选择|选项|answer|option|choice)\s*(?:是|为|is)?\s*[:：]?\s*\*{0,2}\s*([A-Z])(?=\s|$|[.,:;)、，。*])',text,re.IGNORECASE)
 indices={ord(x.upper())-65 for x in matches if 0<=ord(x.upper())-65<n}
 if len(indices)==1:return next(iter(indices)),'explicit_unique_label'
 if len(indices)>1:return None,'ambiguous_labels'
 exact=[i for i,v in enumerate(options) if str(v).strip() and text.strip().rstrip('。.')==str(v).strip().rstrip('。.')]
 if len(exact)==1:return exact[0],'exact_option_text'
 return None,'unresolved'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--records',required=True);a=ap.parse_args()
 p=Path(a.run);src=p/'measurements.jsonl';records={r['id']:r for r in map(json.loads,Path(a.records).read_text().splitlines())};rows=[]
 for r in map(json.loads,src.read_text().splitlines()):
  if r['mode']!='answer_only':continue
  pred,rule=parse(r.get('generated_text',''),records[r['id']]['options'])
  rows.append(dict(key=r['key'],id=r['id'],repeat=r['repeat'],prediction=pred,correct=pred==r['gold'],rule=rule,original_prediction=r['prediction'],truncated=r.get('hit_token_cap',False)))
 raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);(p/'audited-native-answers.jsonl').write_text(raw)
 summary=dict(n=len(rows),correct=sum(r['correct'] for r in rows),unresolved=sum(r['prediction'] is None for r in rows),parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),measurements_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),scope='Deterministic post-hoc formatting parser; gold-blind extraction, no timing or inference changes; not human-adjudicated semantic correctness')
 (p/'native-answer-audit.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
