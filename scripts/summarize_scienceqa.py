"""Audit paired ScienceQA results against the frozen ID/answer manifest."""
import json
import math
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES={'qwen35-2b':'Qwen3.5-2B-Base','gemma-e2b':'Gemma-3n-E2B-it','gemma-e4b':'Gemma-3n-E4B-it','minicpm-v45':'MiniCPM-V-4.5','internvl35-8b':'InternVL3.5-8B','internvl35-14b':'InternVL3.5-14B'}
def main():
    manifest=json.loads((ROOT/'benchmarks/scienceqa-test-100-manifest.json').read_text())
    expected={row['id']:row['expected'] for row in manifest['rows']}
    models=[]
    for key,name in NAMES.items():
        report=json.loads((ROOT/'evidence/scienceqa'/f'{key}.json').read_text())
        assert report['status']=='completed'
        assert report['training_applied'] is False
        assert report['summary']['changed_parameter_version_count']==0
        assert report['suite_sha256']==manifest['suite_sha256']
        assert report['dataset']==manifest['dataset']
        rows=report['results']
        assert len(rows)==len(expected) and {r['id'] for r in rows}==set(expected)
        assert all(r['expected']==expected[r['id']] for r in rows)
        native=sum(r['native_answer']==expected[r['id']] for r in rows)
        decision=sum(r['decision_answer']==expected[r['id']] for r in rows)
        agree=sum(r['native_answer']==r['decision_answer'] for r in rows)
        assert native==report['summary']['native_correct'] and decision==report['summary']['decision_correct'] and agree==report['summary']['agreements']
        assert all(r['finite_pattern_matches'] and r['visual_tokens']>0 for r in rows)
        timings=sorted(r['decision_forward_ms'] for r in rows)
        assert all(math.isfinite(t) and t>0 for t in timings)
        models.append(dict(forward_p50_ms=statistics.median(timings),forward_p95_ms=timings[math.ceil(.95*len(timings))-1],forward_mean_ms=statistics.mean(timings),model_id=key,name=name,transformers_version=report['transformers_version'],torch_version=report['torch_version'],count=len(rows),native_correct=native,decision_correct=decision,agreements=agree,max_full_vocab_abs_diff=max(r['full_vocab_max_abs_diff'] for r in rows),changed_parameter_version_count=report['summary']['changed_parameter_version_count']))
    summary=dict(dataset=manifest['dataset'],suite_id=manifest['suite_id'],suite_sha256=manifest['suite_sha256'],models=models)
    (ROOT/'evidence/scienceqa/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    table='| Model | Native candidate accuracy | Decision accuracy | Difference (pp) | Agreement | Max logit difference |\n|---|---:|---:|---:|---:|---:|\n'
    for r in models:
        n=r['count']
        table+=f"| {r['name']} | {100*r['native_correct']/n:.1f}% | {100*r['decision_correct']/n:.1f}% | {100*(r['decision_correct']-r['native_correct'])/n:+.1f} | {100*r['agreements']/n:.1f}% | {r['max_full_vocab_abs_diff']:g} |\n"
    (ROOT/'docs/scienceqa-results.md').write_text('# ScienceQA paired evaluation\n\n100 fixed image-bearing questions from the official test split, seed 42. Each backbone uses the same subset and paired prepared inputs. Accuracy is scored from candidate-label logits on the official generation first step and the decision forward path using the same softmax and argmax.\n\n'+table+'\n[Method](scienceqa.md) · [Manifest](../benchmarks/scienceqa-test-100-manifest.json) · [Raw records](../evidence/scienceqa/)\n')
    zh=table.replace('Model','模型').replace('Native candidate accuracy','原生候选正确率').replace('Decision accuracy','决策适配正确率').replace('Difference (pp)','差值（百分点）').replace('Agreement','决策一致率').replace('Max logit difference','最大 logits 差异')
    (ROOT/'docs/scienceqa-results_zh.md').write_text('# ScienceQA 基座配对评测结果\n\nScienceQA 官方 test 图像子集固定 100 题，seed 42。六个模型使用同一子集，每组原生对照与决策适配共享已预处理输入。\n\n'+zh+'\n[评测方法](scienceqa_zh.md) · [样本清单](../benchmarks/scienceqa-test-100-manifest.json) · [原始记录](../evidence/scienceqa/)\n')
    latency='| Model | Questions | Median forward (ms) | P95 forward (ms) |\n|---|---:|---:|---:|\n'
    for r in models:
        latency+=f"| {r['name']} | {r['count']} | {r['forward_p50_ms']:.1f} | {r['forward_p95_ms']:.1f} |\n"
    (ROOT/'docs/scienceqa-latency.md').write_text('# Measured decision-forward latency\n\nScienceQA fixed image-test subset, 100 questions per model. NVIDIA A800-SXM4-80GB, BF16, PyTorch 2.8.0, Transformers 4.57.1 for the original five adapters and 5.19.0 for Qwen3.5, shared GPU. Models are loaded before measurement. The synchronized timer covers visual encoding, fusion, and the language-model forward from prepared tensors to returned logits. Each request contains one image and one choice question. Median is the sample median; P95 uses the nearest-rank method.\n\n'+latency+'\n[Raw per-question timings](../evidence/scienceqa/) · [Protocol](scienceqa.md)\n')
    zh_latency=latency.replace('Model','模型').replace('Questions','题数').replace('Median forward (ms)','前向中位数（ms）').replace('P95 forward (ms)','前向 P95（ms）')
    (ROOT/'docs/scienceqa-latency_zh.md').write_text('# 决策前向延迟实测\n\nScienceQA 固定图像测试子集，每模型 100 题。NVIDIA A800-SXM4-80GB、BF16、PyTorch 2.8.0、原五个适配器使用 Transformers 4.57.1，Qwen3.5 使用 5.19.0，共享 GPU 实测。模型加载后，使用 GPU 同步计时；范围为已预处理张量输入到 logits 返回，包含视觉编码、融合与语言模型前向。每次输入一张图像与一个候选选择问题。P95 使用最近秩法。\n\n'+zh_latency+'\n[逐题时间记录](../evidence/scienceqa/) · [评测方法](scienceqa_zh.md)\n')
    print(table)
    print(latency)
if __name__=='__main__': main()
