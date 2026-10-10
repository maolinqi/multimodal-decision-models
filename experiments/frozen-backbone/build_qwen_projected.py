"""Same author selection, projected metadata first; download only selected images.

This avoids transferring all images of excluded two-image NLVR2 records. Selection
still examines every needed record in original file/row order with the same RNG.
"""
import collections, hashlib, io, json, os, random
from pathlib import Path
import fsspec,pyarrow.parquet as pq
from PIL import Image
from build_qwen_data import author,ROOT,REVISION,SOURCE,sha
FILES=json.loads((ROOT/'reference/cauldron_files.json').read_text())
OUT=ROOT/'data/qwen'

def url(p):return f'https://hf-mirror.com/datasets/HuggingFaceM4/the_cauldron/resolve/{REVISION}/{p}'
def open_file(p):return fsspec.open(url(p),'rb',block_size=262144,cache_type='readahead')
def candidates(sub,cap):
 kept=[];offset=0;examined=0;filecounts={}
 for path in sorted(p for p in FILES if p.startswith(sub+'/')):
  with open_file(path) as f:
   pf=pq.ParquetFile(f)
   columns=[pf.schema.column(i).path for i in range(len(pf.schema)) if pf.schema.column(i).path.startswith('texts.') or pf.schema.column(i).path=='images.list.element.path']
   filecounts[path]=pf.metadata.num_rows
   for rg in range(pf.metadata.num_row_groups):
    rows=pf.read_row_group(rg,columns=columns).to_pylist()
    for row_in_group,r in enumerate(rows):
     idx=offset;offset+=1;examined+=1
     if len(r['images'])!=1 or not r['texts']:continue
     t=r['texts'][0];parsed=author.parse(sub,t['user'],t['assistant'])
     if parsed is None:continue
     ctx,q,opts,gold=parsed
     kept.append(dict(source_row=idx,source_file=path,row_group=rg,row_in_group=row_in_group,context='This is a visual question about the image.'+('\n'+ctx if ctx else ''),question=q,options=opts,gold=gold))
     if len(kept)>=cap:return kept,dict(examined=examined,files=filecounts)
   print(sub,path,'metadata examined',examined,'kept',len(kept),flush=True)
 return kept,dict(examined=examined,files=filecounts)
def selected_images(sub,selected):
 groups=collections.defaultdict(list)
 for i,r in enumerate(selected):groups[(r['source_file'],r['row_group'])].append((i,r))
 outputs={}
 for (path,rg),records in groups.items():
  with open_file(path) as f:
   image_rows=pq.ParquetFile(f).read_row_group(rg,columns=['images']).to_pylist()
  for i,r in records:
   image=image_rows[r['row_in_group']]['images'][0]
   if image['bytes'] is None:raise ValueError('Expected embedded image bytes')
   im=Image.open(io.BytesIO(image['bytes']))
   if max(im.size)>768:im=im.copy();im.thumbnail((768,768))
   b=author.png(im);p=OUT/f'{sub}-{i:03d}.png'
   if p.exists():
    if p.read_bytes()!=b:raise ValueError('Previously prepared image differs from reconstructed selection')
   else:p.write_bytes(b)
   outputs[i]=dict(id=f'{sub}:{r["source_row"]}',task=sub,**{k:v for k,v in r.items() if k not in ('row_group','row_in_group')},image=str(p.relative_to(ROOT)),image_sha256=sha(b))
  print(sub,path,'row group',rg,'selected images checked',len(records),flush=True)
 return [outputs[i] for i in range(len(selected))]

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);rng=random.Random(0);counts={};audit={};result=[]
 for sub,(held,_) in author.SUBSETS.items():
  kept,details=candidates(sub,1500 if held else 6000)
  rng.shuffle(kept);n_eval=min(500,len(kept)//5)
  counts[sub]=dict(kept=len(kept),evaluation_pool=n_eval,excluded_from_task_training=held);audit[sub]=details
  print(sub,counts[sub],flush=True)
  if sub in ('raven','visual7w'):result.extend(selected_images(sub,kept[:n_eval][:300]))
  if sub=='visual7w':break
 if len(result)!=600 or len({r['id'] for r in result})!=600:raise ValueError('Expected 600 distinct selected rows')
 payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result)
 tmp=OUT/'records.jsonl.tmp';tmp.write_text(payload);tmp.replace(OUT/'records.jsonl')
 manifest=dict(dataset='HuggingFaceM4/the_cauldron',revision=REVISION,records_sha256=sha(payload.encode()),author_source_sha256=sha(SOURCE.read_bytes()),builder_sha256=sha(Path(__file__).read_bytes()),counts=counts,source_audit=audit,scored_counts={t:sum(r['task']==t for r in result) for t in ('raven','visual7w')},protocol='same original row order, filtering, cap, shared random.Random(0), first 300 evaluation rows; projected metadata is an I/O optimization; original author sample IDs unverified')
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('600 selected rows and images validated',flush=True)
 os._exit(0)
