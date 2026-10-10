"""Render compact README comparison tables from verified public evidence."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def table(headers,rows,textcols=1):
 return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*textcols+['---:']*(len(headers)-textcols))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def section(text,heading,new):
 start=text.index(heading);end=text.find('\n## ',start+len(heading))
 return text[:start]+new.rstrip()+'\n\n'+(text[end+1:] if end>=0 else '')
def main():
 science=read(ROOT/'evidence/scienceqa/summary.json')['models']
 frozen=ROOT/'evidence/frozen-backbone'
 q=read(frozen/'accuracy/qwen/summary.json')['tasks'];g=read(frozen/'accuracy/gemma-normalized-v6/summary.json')['tasks']
 assert all(r['errors']==0 for r in list(q.values())+list(g.values()))
 cases=[('Qwen3.5-2B-Base',f"Visual7W ({q['visual7w']['n']})",'Decider-2B-Vision',89.,q['visual7w']['correct']/q['visual7w']['n']*100),('Gemma-4-26B-A4B-it',None,'Rune v3',75.7,sum(r['correct'] for r in g.values())/sum(r['n'] for r in g.values())*100)]
 for lang,filename in [('en','README.md'),('zh','README_zh.md')]:
  zh=lang=='zh';p=ROOT/filename;text=p.read_text()
  # Remove the old appended Chinese experimental narrative.
  if '### 冻结基座准确率对照' in text:text=text[:text.index('### 冻结基座准确率对照')].rstrip()+'\n'
  accrows=[[model,dataset or ('Rune 公开重建集 (136)' if zh else 'Rune public reconstruction (136)'),trained,f'{reported:.2f}%',f'**{ours:.2f}%**',f'{ours-reported:+.2f}'] for model,dataset,trained,reported,ours in cases]
  acchead='## 免额外训练的实测准确率' if zh else '## Measured accuracy with no additional training'
  acch=['基座','评测集（n）','训练模型','[训练后报告](docs/frozen-backbone-accuracy.md)','本方法（免训练）','差值（pp）'] if zh else ['Backbone','Benchmark (questions)','Trained model','[Reported accuracy](docs/frozen-backbone-accuracy.md)','Ours (0 training)','Delta (pp)']
  scirows=[[r['name'],'0',f"{r['native_correct']/r['count']*100:.0f}%",f"{r['decision_correct']/r['count']*100:.0f}%",f"{r['agreements']}/{r['count']}"] for r in science]
  scih=['基座','额外训练','原生候选准确率','本方法准确率','选择一致'] if zh else ['Backbone','Extra training','Native candidate accuracy','Ours','Agreement']
  acc=acchead+'\n\n'+table(acch,accrows,3)+'\n'+('### ScienceQA（固定 100 题）' if zh else '### ScienceQA (fixed 100 questions)')+'\n\n'+table(scih,scirows)+'\n'+('[准确率记录](docs/frozen-backbone-accuracy.md) · [ScienceQA 协议与结果](docs/scienceqa_zh.md)' if zh else '[Accuracy evidence](docs/frozen-backbone-accuracy.md) · [ScienceQA protocol and results](docs/scienceqa.md)')+'\n'
  text=section(text,acchead,acc)
  lathead='## 低延迟决策：具体用了多久？' if zh else '## Measured decision latency'
  latrows=[]
  for model,name,n in [('gemma','Gemma-4-26B-A4B-it',136)]:
   for kind in ['latency']:
    q=frozen/kind/model/'summary.json'
    label=('自然生成' if zh else 'Natural generation') if kind=='latency' else ('仅回答字母' if zh else 'Single-letter answer')
    if q.exists() and read(q).get('complete'):
     s=read(q);a=s['paired']['answer_only'];assert a['n_questions']==n
     cap_link='docs/identical-input-latency.md#what-was-timed' if kind=='latency' else 'docs/frozen-backbone-accuracy.md#completed-minimal-answer-latency-control'
     marker=f'[†]({cap_link})' if s['modes']['answer_only']['hit_token_cap'] else ''
     latrows.append([name,label,n,f"{s['modes']['answer_only']['generated_tokens_median']:g}",f"{a['native_median_ms']:.1f}{marker}",f"**{a['decision_median_ms']:.1f}**",f"{a['native_median_ms']-a['decision_median_ms']:.1f}",f"**{a['speedup_ratio']:.2f}\u00d7**"])
    else:latrows.append([name,label,n,'—','测试中' if zh else 'Running','测试中' if zh else 'Running','—','—'])
  extra=[]
  for model,name in [('gemma-e4b','Gemma-3n-E4B-it'),('internvl35-8b','InternVL3.5-8B')]:
   path=ROOT/'evidence/instruction-latency'/model/'audited-summary.json'
   if not path.exists():continue
   s=read(path)
   assert s['complete'] and s['n']==100 and s['repeats']==3 and s['records']==700
   assert s['first_step_controls']==100 and s['max_full_vocab_abs_diff']==0 and s['parameter_versions_unchanged']
   marker='[†](docs/instruction-latency.md)' if s['cap_hits'] else ''
   extra.append([name,'自然生成' if zh else 'Natural generation','ScienceQA (100)',f"{s['native_tokens_median']:g}",f"{s['native_median_ms']:.1f}{marker}",f"**{s['decision_median_ms']:.1f}**",f"{s['native_median_ms']-s['decision_median_ms']:.1f}",f"**{s['speedup']:.2f}×**"])
  if extra:
   for row in latrows:row[2]=f'Rune ({row[2]})'
   latrows.extend(extra)
  lath=['基座模型','原模型回答方式','测试集（题数）' if extra else '测试题数','原模型输出长度（token，中位数）','改造前延迟（ms，中位数）','改造后延迟（ms，中位数）','减少的延迟（ms）','加速倍数'] if zh else ['Base model','Original response mode','Test set (questions)' if extra else 'Test questions','Original output length (tokens, median)','Before conversion (ms, median)','After conversion (ms, median)','Latency saved (ms)','Speedup']
  frows=[[r['name'],f"{r['forward_p50_ms']:.1f}",f"{r['forward_p95_ms']:.1f}"] for r in science]
  fh=['基座','前向中位数（ms）','前向 P95（ms）'] if zh else ['Backbone','Median forward (ms)','P95 forward (ms)']
  result_doc='docs/identical-input-latency.md' if (ROOT/'docs/identical-input-latency.md').exists() else 'docs/frozen-backbone-accuracy.md#completed-minimal-answer-latency-control'
  links=f"[{'逐题结果' if zh else 'Per-question results'}]({result_doc})"
  if extra:links+=' · '+('[ScienceQA 延迟对照](docs/instruction-latency.md)' if zh else '[ScienceQA response comparison](docs/instruction-latency.md)')
  latency=lathead+'\n\n'+table(lath,latrows,2)+'\n'+links+'\n\n<details>\n<summary>ScienceQA '+('决策前向耗时' if zh else 'decision-forward timings')+'</summary>\n\n'+table(fh,frows)+'\n'+('[完整结果](docs/scienceqa-latency_zh.md)' if zh else '[Full results](docs/scienceqa-latency.md)')+'\n\n</details>\n'
  text=section(text,lathead,latency)
  text=text.replace('六份完整配对记录','七份完整配对记录').replace('六份逐题报告','七份逐题报告')
  p.write_text(text.rstrip()+'\n')
 print('Rendered comparison tables in the accuracy and latency sections of both READMEs')
if __name__=='__main__':main()
