"""Download the eight published cards, preserving the publisher's asset hashes.

This is a loading/pipeline check, NOT a substitute for Rune's 136-item score.
"""
import hashlib, json, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REV='792e8db555d9b9cf78e187002b6febebdc85fc62'
BASE=f'https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/{REV}/'
CARDS=[
 ('parcel-damaged','Is the parcel visibly damaged?',['yes','no'],0),
 ('receipt-readable','Is the receipt readable?',['yes','no'],0),
 ('excel-format','Goal: change selected cells to type “Text”. Which labelled marker should be clicked?',None,1),
 ('browser-new-tab','Goal: open a new tab. Which labelled marker should be clicked?',None,4),
 ('mobile-translate','Goal: translate. Which labelled marker should be clicked?',None,3),
 ('windows-copy','Goal: copy the file. Which labelled marker should be clicked?',None,0),
 ('geometry-parallelogram','Find the perimeter of the parallelogram.',['78','70.2','93.6','85.8'],0),
 ('finqa-net-revenue','What is the net change in net revenue during 2015 for Entergy Corporation?',['103.4','84.6','94','112.8'],2),
]

if __name__=='__main__':
 out=ROOT/'data/gemma-examples';out.mkdir(parents=True,exist_ok=True)
 manifest=urllib.request.urlopen(BASE+'public/image-jev/examples/manifest.json',timeout=60).read()
 (out/'source-manifest.json').write_bytes(manifest)
 examples={Path(x['asset']).stem:x for x in json.loads(manifest)['examples']}; rows=[]
 for i,(key,question,options,gold) in enumerate(CARDS):
  entry=examples[key];b=urllib.request.urlopen(BASE+'public'+entry['asset'],timeout=60).read()
  sha=hashlib.sha256(b).hexdigest()
  if sha!=entry['asset_sha256']:raise RuntimeError('Published asset hash mismatch')
  p=out/f'image-{i:02d}.webp';p.write_bytes(b)
  rows.append(dict(id=entry['source_item_id'],task='public_example_cards',question=question,options=options or ['Click marker '+x for x in 'ABCDE'],gold=gold,state={},image=str(p.relative_to(ROOT)),image_sha256=sha,source=entry))
 payload=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)
 (out/'records.jsonl').write_text(payload)
 (out/'manifest.json').write_text(json.dumps(dict(count=8,source_revision=REV,records_sha256=hashlib.sha256(payload.encode()).hexdigest(),scope='published WebP example cards; not Rune 136-item replicated set'),indent=2))
 print('Eight published cards downloaded and hashes verified',flush=True)
