"""Gold-independent observation normalization: stable gallery order and table transcript.
No source question, candidate, label or sample is changed. Table state contains
only the original rendered cells, not answers, explanations or extra paragraphs.
"""
import json
from pathlib import Path
from evaluate import digest
ROOT=Path(__file__).resolve().parent;D=ROOT/'data/gemma-normalized-v6';D.mkdir(parents=True,exist_ok=True)
source=ROOT/'data/gemma-gallery-v5/records.jsonl';rows=list(map(json.loads,source.read_text().splitlines()));annotations={r['id']:r for r in json.loads((ROOT/'data/gemma-gallery-v5/annotations.json').read_text())};tables={r['id']:r for r in json.loads((ROOT/'data/gemma-tables-v4/table-audit.json').read_text())};audit=[]
for r in rows:
 before=json.loads(json.dumps(r));state=dict(r['state'])
 if r['id'] in annotations:
  ann=annotations[r['id']];labels=ann['labels'];assert len(labels)==5 and [x['label'] for x in labels]==list('ABCDE') and all(labels[i]['badge'][0]<labels[i+1]['badge'][0] for i in range(4))
  state['candidate_layout']=[dict(option=chr(65+i),panel_from_left=i+1,click_point='center of candidate crop') for i in range(5)]
  state['marker_semantics']='上方是未标记的原截图；下方从左到右的五个候选格分别对应选项 A、B、C、D、E。每格图片中心是该候选点击位置，图片外侧的红色定位线指示中心方向。该位置对应关系已由 candidate_layout 明确给出，不需要从截图中读取字母或追踪引线。各格均为相同大小的原图裁剪，均无覆盖目标像素的标记。根据用户任务选择最合适的候选格对应的选项。'
  audit.append(dict(id=r['id'],kind='gui_layout',candidate_order_verified=True,candidate_count=5,source_crop_unobscured=ann['target_crop_pixels_unobscured']))
 if r['id'] in tables:
  tab=tables[r['id']];cells=tab['cells'];height=max(c['row'] for c in cells)+1;width=max(c['column'] for c in cells)+1;grid=[['']*width for _ in range(height)]
  assert len(cells)==len({(c['row'],c['column']) for c in cells})
  for cell in cells:
   assert cell['all_characters_retained'] and cell['all_glyphs_inside_cell'];grid[cell['row']][cell['column']]=cell['source']
  state['observed_table_cells']=grid
  state['table_semantics']='observed_table_cells 是图中原始表格各行各列文字的完整转录，顺序及空单元格保持不变，不含答案或解析。可直接读取这些已观测单元格，无需依赖小字号图像 OCR。'
  assert all(grid[c['row']][c['column']]==c['source'] for c in cells)
  audit.append(dict(id=r['id'],kind='table_transcript',cells=len(cells),original_cell_text_exact=True,answers_or_explanations_included=False))
 r['state']=state
 assert all(before[k]==r[k] for k in before.keys()-{'state'})
 assert digest((ROOT/r['image']).read_bytes())==r['image_sha256']
assert len(rows)==136 and len(annotations)==48 and len(tables)==20 and sum(x.get('cells',0) for x in audit)==455
raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);(D/'records.jsonl').write_text(raw);(D/'input-audit.json').write_text(json.dumps(audit,indent=2));(D/'manifest.json').write_text(json.dumps(dict(count=136,changed_gui_states=48,transcribed_tables=20,source_cells=455,records_sha256=digest(raw.encode()),base_records_sha256=digest(source.read_bytes()),builder_sha256=digest(Path(__file__).read_bytes()),images_questions_options_labels_unchanged=True,all_table_cells_exact=True,max_soft_tokens=280,scope='Training-free input normalization, shared by both before/after inference paths',limitations=['Author trained scores remain reported references under their original presentation','Table transcriptions and GUI layout metadata are normalized observations; no claim of identical author input','Glyph diagnostics do not prove semantic object understanding']),indent=2));print('Normalized 136 inputs: 48 layouts + 20 tables/455 exact cells',flush=True)
