"""Gold-independent candidate gallery: unchanged screenshot + equal-sized candidate crops.
No labels, arrows or lines are drawn over screenshot pixels. Every click point
uses the same fixed 256x256 source crop, centered in a uniformly formatted tile.
"""
import json,os,math
from pathlib import Path
from datasets import load_dataset
from PIL import Image,ImageDraw
from transformers import AutoProcessor
from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
from download_models import MODELS
import build_gemma_reconstruction as v1
ROOT=v1.ROOT;OUT=ROOT/'data/gemma-gallery-v5';OUT.mkdir(parents=True,exist_ok=True)
proc=AutoProcessor.from_pretrained(MODELS['gemma'][2],local_files_only=True).image_processor
annotations=[]
def gallery(im,boxes,gold,seed,crop=None):
 # Gold/seed do not affect layout, crop extent, labels, salience or rendering.
 screen=im.convert('RGB').crop(crop) if crop else im.convert('RGB').copy();dx,dy=crop[:2] if crop else (0,0)
 top=screen.copy();top.thumbnail((1024,650));W=1024;header=48;Y=header+top.height+16;H=Y+320
 canvas=Image.new('RGB',(W,H),'white');canvas.paste(top,((W-top.width)//2,header));draw=ImageDraw.Draw(canvas)
 draw.text((20,8),'Original screen',font=v1.font(28),fill='black')
 labels=[]
 for i,b in enumerate(boxes):
  cx=(b[0]+b[2])/2-dx;cy=(b[1]+b[3])/2-dy
  # Same source extent for all candidates; out-of-frame pixels are neutral padding.
  x0=round(cx)-128;y0=round(cy)-128;tile=Image.new('RGB',(256,256),'white')
  region=(max(0,x0),max(0,y0),min(screen.width,x0+256),min(screen.height,y0+256))
  tile.paste(screen.crop(region),(region[0]-x0,region[1]-y0));tile=tile.resize((176,176),Image.Resampling.BICUBIC)
  left=round(i*W/5);right=round((i+1)*W/5);mx=(left+right)//2;px=mx-88;py=Y+36
  canvas.paste(tile,(px,py));draw.rectangle((px-2,py-2,px+177,py+177),outline='#b00020',width=2)
  # Edge ticks indicate the tile center without covering any screenshot pixel.
  draw.line((mx,Y+20,mx,Y+31),fill='#b00020',width=4);draw.line((mx,py+181,mx,py+192),fill='#b00020',width=4)
  draw.line((px-12,py+88,px-5,py+88),fill='#b00020',width=4);draw.line((px+181,py+88,px+188,py+88),fill='#b00020',width=4)
  ly=Y+267;draw.text((mx,ly),v1.LABELS[i],font=v1.font(88),anchor='mm',fill='black')
  # The pasted source crop is byte-for-byte untouched by all annotation drawing.
  if canvas.crop((px,py,px+176,py+176)).tobytes()!=tile.tobytes():raise ValueError('Gallery annotation obscures source crop')
  labels.append(dict(label=v1.LABELS[i],source_target=[cx,cy],target=[mx,py+88],badge=[mx,ly],bbox=[left,Y+220,right,Y+318],tile_bbox=[px,py,px+176,py+176]))
 hh,ww=get_aspect_ratio_preserving_size(H,W,proc.patch_size,280*proc.pooling_kernel_size**2,proc.pooling_kernel_size)
 scale=min(ww/W,hh/H)
 if 88*scale<28:raise ValueError('Processed label too small')
 annotations.append(dict(source_size=[W,H],source_screen_size=list(screen.size),processed_size=[ww,hh],labels=labels,font_processed_px=88*scale,source_crop_extent=256,target_crop_pixels_unobscured=True,original_screen_unmarked=True,crop=crop))
 return canvas
v1.markers=gallery
items=json.loads((ROOT/'reference/gemma_preview_items.json').read_text());sources=json.loads((ROOT/'reference/gemma_source_files.json').read_text());rows=list(map(json.loads,(ROOT/'data/gemma-tables-v4/records.jsonl').read_text().splitlines()));lookup={r['id']:r for r in rows}
for family in ('ScreenSpot','Multimodal-Mind2Web'):
 chosen={r['source_row']:r for r in items if r['dataset']==family};src=sources[family];files=src['files'][:1] if family=='Multimodal-Mind2Web' else src['files']
 ds=load_dataset('parquet',data_files={'train':[f'https://hf-mirror.com/datasets/{src["repo"]}/resolve/{src["revision"]}/{p}' for p in files]},split='train',streaming=True)
 for index,row in enumerate(ds):
  if index>max(chosen):break
  if index not in chosen:continue
  item=chosen[index];im=v1.gui_screen(row,item) if family=='ScreenSpot' else v1.gui_web(row,item);ident='preview:'+item['id'];annotations[-1]['id']=ident
  p=OUT/(item['id']+'.png');im.save(p);r=lookup[ident];r['image']=str(p.relative_to(ROOT));r['image_sha256']=v1.sha(p.read_bytes());r['rendering']='gallery v5; unmarked original screen plus five uniform candidate-center crops'
  r['state']=dict(r['state'],marker_semantics='上方是未加标记的原截图。下方 A～E 各格展示一个候选点击位置周围相同大小的原图区域；所有格子的实际候选点击点都是该格图片的中心，红色边缘定位线指示中心方向，不遮挡图像。格子下方的大写字母是选项编号，不是功能名称或置信度。依据用户任务选择点击点对应的格子字母。所有候选使用相同裁剪大小和样式。')
  print(ident,'processed font',annotations[-1]['font_processed_px'],flush=True)
assert len(annotations)==48
old=list(map(json.loads,(ROOT/'data/gemma-tables-v4/records.jsonl').read_text().splitlines()))
for a,b in zip(old,rows):assert all(a.get(k)==b.get(k) for k in a.keys()-{'image','image_sha256','rendering','state'})
raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);(OUT/'records.jsonl').write_text(raw);(OUT/'annotations.json').write_text(json.dumps(annotations,indent=2));(OUT/'manifest.json').write_text(json.dumps(dict(count=136,changed_gui_images=48,records_sha256=v1.sha(raw.encode()),base_records_sha256=v1.sha((ROOT/'data/gemma-tables-v4/records.jsonl').read_bytes()),builder_sha256=v1.sha(Path(__file__).read_bytes()),target_crops_unobscured=True,labels_and_selection_unchanged=True,max_soft_tokens=280,minimum_processed_font_px=min(a['font_processed_px'] for a in annotations),limitations=['Gallery adaptation is not author-pixel replication','Source GUI distractor points and crop remain reconstructed','Glyph and association recognition must be measured separately; geometric checks are insufficient']),indent=2));print('Gallery v5 built',flush=True)
os._exit(0)
