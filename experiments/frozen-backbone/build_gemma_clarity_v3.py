"""Marker-only input ablation. Never changes selection, labels, crop or v1 pixels."""
import io,json,math,os
from pathlib import Path
from datasets import load_dataset
from PIL import Image,ImageDraw,ImageChops
import build_gemma_reconstruction as v1
import torch
torch.set_num_threads(4)
from download_models import MODELS
from transformers import AutoProcessor
from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
ROOT=v1.ROOT; OUT=ROOT/'data/gemma-clarity-v3'; OUT.mkdir(parents=True,exist_ok=True)
processor=AutoProcessor.from_pretrained(MODELS['gemma'][2],local_files_only=True).image_processor
annotations=[]
def intersects(a,b):return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]
def clear_markers(im,boxes,gold,seed,crop=None):
 # gold/seed deliberately do not affect style or placement.
 im=im.convert('RGB'); im=im.crop(crop) if crop else im.copy(); original=im.copy(); w,h=im.size
 dx,dy=crop[:2] if crop else (0,0)
 hh,ww=get_aspect_ratio_preserving_size(h,w,processor.patch_size,280*processor.pooling_kernel_size**2,processor.pooling_kernel_size)
 scale=min(ww/w,hh/h); radius=math.ceil(22/scale); fs=math.ceil(28/scale); stroke=max(2,math.ceil(2/scale))
 targets=[(b[0]-dx,b[1]-dy,b[2]-dx,b[3]-dy) for b in boxes]; used=[]; placed=[]
 protected=[(max(0,math.floor(b[0])-3),max(0,math.floor(b[1])-3),min(w,math.ceil(b[2])+3),min(h,math.ceil(b[3])+3)) for b in targets]
 for i,b in enumerate(targets):
  cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2; choices=[]
  # Uniform grid plus edges of every target gives deterministic placement without hiding any candidate box.
  step=max(1,radius)
  for x in range(radius+stroke,w-radius-stroke,step):
   for y in range(radius+stroke,h-radius-stroke,step):
    bb=(x-radius-stroke,y-radius-stroke,x+radius+stroke,y+radius+stroke)
    if any(intersects(bb,t) for t in protected+used):continue
    choices.append(((x-cx)**2+(y-cy)**2,y,x,bb))
  if not choices:raise ValueError('No non-overlapping badge position; needs manual input review')
  _,y,x,bb=min(choices);used.append(bb);placed.append((cx,cy,x,y))
 d=ImageDraw.Draw(im)
 for cx,cy,x,y in placed:
  d.line((cx,cy,x,y),fill='white',width=stroke*3);d.line((cx,cy,x,y),fill='#bd001c',width=stroke)
 # Hollow outlines identify target boxes; all drawing inside protected target boxes is removed below.
 for b in protected:
  d.rectangle((max(0,b[0]-stroke),max(0,b[1]-stroke),min(w-1,b[2]+stroke),min(h-1,b[3]+stroke)),outline='#bd001c',width=stroke)
 for i,(cx,cy,x,y) in enumerate(placed):
  d.ellipse((x-radius,y-radius,x+radius,y+radius),fill='white',outline='#bd001c',width=stroke)
  d.text((x,y),v1.LABELS[i],font=v1.font(fs),anchor='mm',fill='black',stroke_width=0)
 # Restore every protected target rectangle EXACTLY, including arrow glyphs and button text.
 for b in protected:im.paste(original.crop(b),(b[0],b[1]))
 for b in protected:
  if ImageChops.difference(im.crop(b),original.crop(b)).getbbox() is not None:raise ValueError('Target pixels changed')
 annotations.append(dict(target_pixels_unchanged=True,protected_boxes=protected,source_size=[w,h],processed_size=[ww,hh],max_soft_tokens=280,font_source_px=fs,font_processed_px=fs*scale,diameter_processed_px=2*radius*scale,crop=crop,targets=targets,labels=[dict(label=v1.LABELS[i],target=[cx,cy],badge=[x,y],bbox=used[i]) for i,(cx,cy,x,y) in enumerate(placed)],badge_target_overlap=False,badge_badge_overlap=False))
 return im
v1.markers=clear_markers
items=json.loads((ROOT/'reference/gemma_preview_items.json').read_text());sources=json.loads((ROOT/'reference/gemma_source_files.json').read_text())
records=[json.loads(x) for x in (ROOT/'data/gemma-reconstructed/records.jsonl').read_text().splitlines()]
lookup={r['id']:r for r in records}
for family in ('ScreenSpot','Multimodal-Mind2Web'):
 chosen={r['source_row']:r for r in items if r['dataset']==family};src=sources[family]
 files=src['files'][:1] if family=='Multimodal-Mind2Web' else src['files']
 urls=[f'https://hf-mirror.com/datasets/{src["repo"]}/resolve/{src["revision"]}/{p}' for p in files]
 ds=load_dataset('parquet',data_files={'train':urls},split='train',streaming=True)
 for index,row in enumerate(ds):
  if index>max(chosen):break
  if index not in chosen:continue
  item=chosen[index];im=v1.gui_screen(row,item) if family=='ScreenSpot' else v1.gui_web(row,item)
  path=OUT/(item['id']+'.png');im.save(path);r=lookup['preview:'+item['id']]
  r['image']=str(path.relative_to(ROOT));r['image_sha256']=v1.sha(path.read_bytes());r['rendering']='clarity v3; large offset badges, hollow target boundaries, protected target pixels unchanged'
  r['state']=dict(r['state'],marker_semantics='A～E 是候选点击位置的编号，不是功能名称或置信度。白底红边圆圈内的字母对应一个选项；红色引线连接到红色空心目标框，实际点击位置是该目标框的中心，不是字母圆圈的中心。目标框内部保留原图内容。根据用户任务选择目标框对应的字母。干扰位置也采用同样样式。')
  annotations[-1]['id']=r['id']
  # Show at actual processor resolution, not only original screenshot size.
  processed=processor(im,return_tensors='pt',do_rescale=False,max_soft_tokens=280)
  pv=processed['pixel_values'][0];pp=processor.patch_size;ww,hh=annotations[-1]['processed_size'];n=(ww//pp)*(hh//pp)
  arr=pv[:n].reshape(hh//pp,ww//pp,pp,pp,3).permute(0,2,1,3,4).reshape(hh,ww,3).clamp(0,255).byte().numpy()
  Image.fromarray(arr).save(OUT/(item['id']+'.processed.png'))
  print(r['id'],annotations[-1]['processed_size'],'font processed',annotations[-1]['font_processed_px'],flush=True)
assert len(annotations)==48
for old,new in zip([json.loads(x) for x in (ROOT/'data/gemma-reconstructed/records.jsonl').read_text().splitlines()],records):
 assert all(old.get(k)==new.get(k) for k in old.keys()-{'image','image_sha256','rendering','state'})
payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records);(OUT/'records.jsonl').write_text(payload)
(OUT/'annotations.json').write_text(json.dumps(annotations,indent=2))
(OUT/'manifest.json').write_text(json.dumps(dict(count=136,changed_images=48,marker_only=False,explicit_marker_legend=True,target_pixel_preservation=True,base_records_sha256=v1.sha((ROOT/'data/gemma-reconstructed/records.jsonl').read_bytes()),records_sha256=v1.sha(payload.encode()),builder_sha256=v1.sha(Path(__file__).read_bytes()),v1_builder_sha256=v1.sha(Path(v1.__file__).read_bytes()),minimum_processed_font_px=min(x['font_processed_px'] for x in annotations),max_soft_tokens=280,limitations=['Geometric legibility checks do not prove model recognition','Pointers and larger markers are an input ablation; author pixels still not reproduced','Public cards and non-GUI inputs unchanged; existing table clipping is a separate unresolved factor']),indent=2))
print('Clarity v3 complete; all 136 labels/questions/selection unchanged; target pixels preserved',flush=True)
os._exit(0)
