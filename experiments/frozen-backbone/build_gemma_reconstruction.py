"""Rebuild preview source rows plus published cards, with explicit rendering caveats.

Upstream identities and labels are verified. GUI marker placement, crop and table
rendering are newly deterministic, not claimed to reproduce the author's pixels.
"""
import hashlib, io, json, os, random, textwrap
from decimal import Decimal, InvalidOperation
from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/gemma-reconstructed'
LABELS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical_answer(value):
 s=str(value).strip().casefold()
 try:return ('numeric',Decimal(s.removesuffix('%').replace(',','')),s.endswith('%'))
 except InvalidOperation:return ('text',s)
def font(n):
 p='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 return ImageFont.truetype(p,n) if Path(p).exists() else ImageFont.load_default(size=n)
def table_image(table):
 # Original values only: answer, explanation and surrounding gold annotations excluded.
 table=[[str(v) for v in r] for r in table];cols=max(map(len,table))
 widths=[min(750,max(130,max(len(r[j]) if j<len(r) else 0 for r in table)*8+24)) for j in range(cols)]
 wrapped=[[textwrap.wrap(r[j],max(8,(widths[j]-20)//8)) or [''] if j<len(r) else [''] for j in range(cols)] for r in table]
 heights=[max(map(len,r))*22+12 for r in wrapped];im=Image.new('RGB',(sum(widths),sum(heights)), 'white');d=ImageDraw.Draw(im)
 y=0
 for i,row in enumerate(wrapped):
  x=0
  for j,lines in enumerate(row):
   d.rectangle((x,y,x+widths[j]-1,y+heights[i]-1),fill='#eeeeee' if i==0 else 'white',outline='#aaaaaa')
   d.multiline_text((x+8,y+5),'\n'.join(lines),font=font(16),fill='black',spacing=2);x+=widths[j]
  y+=heights[i]
 return im
def rect(candidate):
 c=json.loads(candidate) if isinstance(candidate,str) else candidate
 attrs=json.loads(c['attributes']) if isinstance(c['attributes'],str) else c['attributes']
 x,y,w,h=map(float,attrs['bounding_box_rect'].split(','))
 return c['backend_node_id'],(x,y,x+w,y+h)
def markers(im,boxes,gold,seed,crop=None):
 im=im.convert('RGB')
 if crop:im=im.crop(crop)
 dx,dy=(crop[0],crop[1]) if crop else (0,0)
 im=im.copy();d=ImageDraw.Draw(im)
 for i,b in enumerate(boxes):
  x1,y1,x2,y2=b;x=(x1+x2)/2-dx;y=(y1+y2)/2-dy
  radius=12
  # Every label has exactly the same style; gold is not made visually salient.
  d.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#ffef45',outline='black',width=2)
  d.text((x,y),LABELS[i],font=font(17),anchor='mm',fill='black')
 return im
def gui_screen(row,item):
 im=row['image'];w,h=im.size
 bbox=item['source_label']['bbox_xyxy'];actual=row['bbox']
 if len(actual)!=4 or any(abs(a-b)>1e-5 for a,b in zip(actual,bbox)):raise ValueError('ScreenSpot bbox mismatch')
 if row['file_name']!=item['source_label']['file_name']:raise ValueError('ScreenSpot source image mismatch')
 target=tuple(v*(w if i%2==0 else h) for i,v in enumerate(bbox));rng=random.Random(item['source_row'])
 boxes=[]
 while len(boxes)<4:
  x=rng.uniform(20,w-20);y=rng.uniform(20,h-20)
  if target[0]-20<x<target[2]+20 and target[1]-20<y<target[3]+20:continue
  if any((x-(b[0]+b[2])/2)**2+(y-(b[1]+b[3])/2)**2<40**2 for b in boxes):continue
  boxes.append((x-1,y-1,x+1,y+1))
 gold=list(item['rubric']['criteria']).index(item['gold']);boxes.insert(gold,target)
 return markers(im,boxes,gold,item['source_row'])
def gui_web(row,item):
 if row['action_uid']!=item['source_label']['action_uid']:raise ValueError('Mind2Web action identity mismatch')
 im=row['screenshot'];w,h=im.size
 targetid=item['source_label']['positive_backend_node_id']
 positive=[b for c in row['pos_candidates'] for ident,b in [rect(c)] if str(ident)==str(targetid)]
 if len(positive)!=1:raise ValueError('Mind2Web positive identity mismatch')
 target=positive[0];cy=(target[1]+target[3])/2
 top=max(0,min(h-800,int(cy-400)));bottom=min(h,top+800);crop=(0,top,w,bottom)
 boxes=[]
 for c in row['neg_candidates']:
  try:_,b=rect(c)
  except (ValueError,KeyError):continue
  x=(b[0]+b[2])/2;y=(b[1]+b[3])/2
  if b[2]-b[0]>w/2 or b[3]-b[1]>300 or b[2]<=b[0] or b[3]<=b[1]:continue
  if not (15<x<w-15 and top+15<y<bottom-15):continue
  if target[0]-20<x<target[2]+20 and target[1]-20<y<target[3]+20:continue
  if any((x-(z[0]+z[2])/2)**2+(y-(z[1]+z[3])/2)**2<30**2 for z in boxes):continue
  boxes.append(b)
  if len(boxes)==4:break
 if len(boxes)!=4:raise ValueError('Cannot reconstruct four visible original negative elements')
 gold=list(item['rubric']['criteria']).index(item['gold']);boxes.insert(gold,target)
 return markers(im,boxes,gold,item['source_row'],crop)

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 items=json.loads((ROOT/'reference/gemma_preview_items.json').read_text())
 sources=json.loads((ROOT/'reference/gemma_source_files.json').read_text());records=[]
 for family,source in sources.items():
  chosen={r['source_row']:r for r in items if r['dataset']==family};print('Reconstructing',family,len(chosen),flush=True)
  cache=OUT/(family+'.records.json')
  if cache.exists():
   saved=json.loads(cache.read_text())
   if saved['builder_sha256']==sha(Path(__file__).read_bytes()) and saved['source']==source and all(sha((ROOT/r['image']).read_bytes())==r['image_sha256'] for r in saved['records']):
    records.extend(saved['records']);print('Reusing verified family',family,flush=True);continue
  files=source['files'][:1] if family in ('CLEVR-HOPE','ArxivQA','Multimodal-Mind2Web') else source['files']
  urls=[f'https://hf-mirror.com/datasets/{source["repo"]}/resolve/{source["revision"]}/{p}' for p in files]
  ds=load_dataset('parquet',data_files={'train':urls},split='train',streaming=True)
  for index,row in enumerate(ds):
   if index not in chosen:
    if index>max(chosen):break
    continue
   item=chosen[index];gold=item['gold'];question=item['rubric']['instructions']
   if family=='CLEVR-HOPE':
    if row['query']!=question or str(row['answer']).lower()!=gold:raise ValueError('CLEVR row/label mismatch')
    im=row['image']
   elif family=='Geometry3K':
    if row['problem'].replace('<image>','').strip()!=question or str(row['answer'])!=item['rubric']['criteria'][gold]:raise ValueError('Geometry row/label mismatch')
    im=row['images'][0]
   elif family=='ArxivQA':
    user=json.loads(row['messages'])[0]
    if user['question']!=question or row['answer']!=gold:raise ValueError('Arxiv row/label mismatch')
    im=row['media'][0]
   elif family=='FinQA':
    if row['question']!=question or canonical_answer(row['answer'])!=canonical_answer(item['rubric']['criteria'][gold]):raise ValueError('FinQA row/label mismatch')
    im=table_image(row['table_ori'])
   elif family=='ScreenSpot':im=gui_screen(row,item)
   else:im=gui_web(row,item)
   b=io.BytesIO();im.convert('RGB').save(b,format='PNG');b=b.getvalue();p=OUT/(item['id']+'.png');p.write_bytes(b)
   records.append(dict(id='preview:'+item['id'],task=family,question=question,options=list(item['rubric']['criteria'].values()),gold=list(item['rubric']['criteria']).index(gold),state={k:v for k,v in item['state'].items() if k!='image'},image=str(p.relative_to(ROOT)),image_sha256=sha(b),source_revision=source['revision'],source_row=index,rendering='original image' if family in ('CLEVR-HOPE','Geometry3K','ArxivQA') else 'new deterministic rendering; author pixels not verified'))
   print(item['id'],'source identity verified',flush=True)
  if sum(r['task']==family for r in records)!=len(chosen):raise ValueError('Incomplete source family')
  cache.write_text(json.dumps(dict(builder_sha256=sha(Path(__file__).read_bytes()),source=source,records=[r for r in records if r['task']==family]),ensure_ascii=False))
 cards=[json.loads(x) for x in (ROOT/'data/gemma-examples/records.jsonl').read_text().splitlines()]
 for card in cards:card['id']='card:'+card['id'];records.append(card)
 if len(records)!=136 or len({r['id'] for r in records})!=136:raise ValueError('Expected 136 distinct sample occurrences')
 payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records);(OUT/'records.jsonl').write_text(payload)
 (OUT/'manifest.json').write_text(json.dumps(dict(count=136,source_files=sources,records_sha256=sha(payload.encode()),scope='128 preview original rows + eight published cards',limitations=['Author 136-item identities/renderings not verified','Table renderer, GUI crop and negative-marker selection reconstructed; approximate comparison only','Published card WebPs differ from original source pixels']),indent=2))
 print('All 136 reconstructed occurrences saved and checked',flush=True)
 # Streaming Arrow workers can crash at interpreter teardown after all work is done.
 # Exit only after the complete manifest and all records have been written/validated.
 os._exit(0)
