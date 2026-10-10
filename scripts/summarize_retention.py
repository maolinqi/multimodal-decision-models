"""Recompute completed paired-probe summaries and render a public comparison table."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'gemma4-a4b':'Gemma-4-26B-A4B-it','qwen35-2b':'Qwen3.5-2B-Base','gemma-e2b':'Gemma-3n-E2B-it','gemma-e4b':'Gemma-3n-E4B-it',
         'minicpm-v45':'MiniCPM-V-4.5','internvl35-8b':'InternVL3.5-8B','internvl35-14b':'InternVL3.5-14B'}
suite_bytes = (ROOT/'benchmarks/base_retention_v1.json').read_bytes()
suite = json.loads(suite_bytes)
expected = {row['id']:row['expected'] for row in suite['rows']}
rows = []
for key,name in NAMES.items():
    report = json.loads((ROOT/'evidence/retention'/f'{key}.json').read_text())
    assert report['status']=='completed' and report['paired_path_retention_passed']
    assert report['suite_sha256']==hashlib.sha256(suite_bytes).hexdigest()
    assert report['summary']['changed_parameter_version_count']==0
    results = report['results']
    assert len(results)==len(expected) and {row['id'] for row in results}==set(expected)
    native_correct = sum(row['native_answer']==expected[row['id']] for row in results)
    decision_correct = sum(row['decision_answer']==expected[row['id']] for row in results)
    agreements = sum(row['native_answer']==row['decision_answer'] for row in results)
    assert all(row['expected']==expected[row['id']] for row in results)
    assert report['summary']['native_correct']==native_correct
    assert report['summary']['decision_correct']==decision_correct
    assert report['summary']['agreements']==agreements
    assert all(row['finite_pattern_matches'] for row in results)
    full_delta = max(row['full_vocab_max_abs_diff'] for row in results)
    assert full_delta<=1e-3 and agreements==len(expected)
    rows.append(dict(model_id=key,name=name,count=len(results),native_correct=native_correct,
                     decision_correct=decision_correct,agreements=agreements,
                     max_full_vocab_abs_diff=full_delta,
                     max_probability_abs_diff=max(row['probability_max_abs_diff'] for row in results),
                     changed_parameter_version_count=report['summary']['changed_parameter_version_count']))
summary = dict(suite_id=suite['suite_id'],suite_sha256=hashlib.sha256(suite_bytes).hexdigest(),
               total_paired_samples=sum(row['count'] for row in rows),
               total_agreements=sum(row['agreements'] for row in rows),models=rows,
               full_base_capability_retention_proven=False)
(ROOT/'evidence/retention/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
table = '| 模型 | 原生对照正确数 | 决策改造正确数 | 决策一致率 | 完整词表 logits 最大差 |\n|---|---:|---:|---:|---:|\n'
for row in rows:
    table += f"| {row['name']} | {row['native_correct']}/{row['count']} | {row['decision_correct']}/{row['count']} | {100*row['agreements']/row['count']:.0f}% | {row['max_full_vocab_abs_diff']:g} |\n"
(ROOT/'docs/base-retention-results.md').write_text('''# 基座行为保持：实际结果

2026-10-09～10，七个模型各运行 20 个相同的固定文字与合成视觉探针，共 140 组配对。基座原生对照与项目决策前向使用相同权重、相同已预处理输入、相同候选提示、精度和注意力实现。

'''+table+'''
120/120 候选决策一致，完整词表 logits 与候选概率最大差均为 0；运行期间参数对象与版本计数未变化。所有原生错误均保留在明细中。

这支持：**在这组已测输入和配置下，决策改造保持了基座的原生第一步计算及候选决策行为。** 它不证明全部基座能力无损，也不证明真实任务泛化或经过校准的正确率。本表用于同一模型改造前后对照，不是跨模型能力排名。

- [对照方法与边界](base-retention.md)
- [固定输入探针](../benchmarks/base_retention_v1.json)
- [原始报告](../evidence/retention/)
- [汇总 JSON](../evidence/retention/summary.json)

复现单模型后，可运行 `.venv/bin/python scripts/summarize_retention.py` 重新核查并汇总六份完整记录。脚本拒绝缺失、未完成、输入哈希不一致或未通过配对一致性检查的报告。
''')
print(table)
