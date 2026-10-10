"""Table-only ablation on v3 inputs; pixel-measured wrap, no source text truncation."""
import hashlib,json,os
from pathlib import Path
from datasets import load_dataset
from PIL import Image,ImageDraw
import build_gemma_reconstruction as v1
ROOT=v1.ROOT;OUT=ROOT/'data/gemma-tables-v4';OUT.mkdir(parents=True,exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def wrap_pixels(text,font,width):
 lines=[]
 for paragraph in text.split('\n'):
  remainder=paragraph
  if not remainder:lines.append('');continue
  while remainder:
   n=0
   for k in range(1,len(remainder)+1):
    box=font.getbbox(remainder[:k])
    if font.getlength(remainder[:k])>width or box[2]-box[0]>width:break
    n=k
   if n==0:raise ValueError('Cell too narrow for a character')
   if n<len(remainder):
    boundary=remainder[:n].rfind(' ')
    if boundary>=0:n=boundary+1
   lines.append(remainder[:n]);remainder=remainder[n:]
 assert ''.join(lines)==text.replace('\n',''),'Source characters lost'
 return lines

def render(table):
 table=[[str(v) for v in row] for row in table];cols=max(map(len,table));f=v1.font(16);pad=8;line_height=22
 widths=[]
 for j in range(cols):
  maximum=0
  for row in table:
   for line in (row[j] if j<len(row) else '').split('\n'):
    bb=f.getbbox(line);maximum=max(maximum,f.getlength(line),bb[2]-bb[0])
  widths.append(int(min(750,max(130,maximum+2*pad+4))))
 wrapped=[[wrap_pixels(row[j] if j<len(row) else '',f,widths[j]-2*pad-4) for j in range(cols)] for row in table]
 heights=[max(map(len,row))*line_height+2*pad for row in wrapped]
 im=Image.new('RGB',(sum(widths),sum(heights)),'white');d=ImageDraw.Draw(im);audit=[];y=0
 for i,row in enumerate(wrapped):
  x=0
  for j,lines in enumerate(row):
   cell=(x,y,x+widths[j],y+heights[i]);d.rectangle((x,y,cell[2]-1,cell[3]-1),fill='#eeeeee' if i==0 else 'white',outline='#aaaaaa')
   boxes=[]
   for k,line in enumerate(lines):
    at=(x+pad,y+pad+k*line_height);box=d.textbbox(at,line,font=f,anchor='lt')
    assert cell[0]<=box[0]<=box[2]<=cell[2]-pad and cell[1]<=box[1]<=box[3]<=cell[3]-pad,(line,box,cell)
    d.text(at,line,font=f,fill='black',anchor='lt');boxes.append(box)
   audit.append(dict(row=i,column=j,source=table[i][j] if j<len(table[i]) else '',lines=lines,cell_bbox=cell,text_bboxes=boxes,all_characters_retained=True,all_glyphs_inside_cell=True));x+=widths[j]
  y+=heights[i]
 return im,audit

if __name__=='__main__':
 items=json.loads((ROOT/'reference/gemma_preview_items.json').read_text());chosen={x['source_row']:x for x in items if x['dataset']=='FinQA'}
 source=json.loads((ROOT/'reference/gemma_source_files.json').read_text())['FinQA'];urls=[f'https://hf-mirror.com/datasets/{source["repo"]}/resolve/{source["revision"]}/{p}' for p in source['files']]
 records=list(map(json.loads,(ROOT/'data/gemma-clarity-v3/records.jsonl').read_text().splitlines()));lookup={x['id']:x for x in records};audit=[]
 ds=load_dataset('parquet',data_files={'train':urls},split='train',streaming=True)
 for i,row in enumerate(ds):
  if i>max(chosen):break
  if i not in chosen:continue
  item=chosen[i]
  assert row['question']==item['rubric']['instructions']
  assert v1.canonical_answer(row['answer'])==v1.canonical_answer(item['rubric']['criteria'][item['gold']])
  im,cells=render(row['table_ori']);path=OUT/(item['id']+'.png');im.save(path);r=lookup['preview:'+item['id']]
  r.update(image=str(path.relative_to(ROOT)),image_sha256=sha(path.read_bytes()),rendering='v4 table-only fix: pixel-measured wrapping, all source characters retained, all glyph boxes inside cells')
  audit.append(dict(id=r['id'],source_row=i,image_size=im.size,cells=cells));print(r['id'],'table glyph bounds and source text verified',flush=True)
 assert len(audit)==20
 old=list(map(json.loads,(ROOT/'data/gemma-clarity-v3/records.jsonl').read_text().splitlines()))
 for a,b in zip(old,records):assert all(a.get(k)==b.get(k) for k in a.keys()-{'image','image_sha256','rendering'})
 raw=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records);(OUT/'records.jsonl').write_text(raw);(OUT/'table-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 (OUT/'manifest.json').write_text(json.dumps(dict(count=136,changed_tables=20,source=source,records_sha256=sha(raw.encode()),base_records_sha256=sha((ROOT/'data/gemma-clarity-v3/records.jsonl').read_bytes()),builder_sha256=sha(Path(__file__).read_bytes()),table_characters_retained=True,table_glyphs_within_cells=True,font_source_px=16,limitations=['Does not guarantee OCR after 280-token downsampling','Author table pixels and prompts still unverified','Public diagnostic set already exposed; not blind test']),indent=2))
 print('Table-only v4 complete; no question, answer, state or GUI changes',flush=True);os._exit(0)
