"""Audit paired ScienceQA results against the frozen ID/answer manifest."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES={'gemma-e2b':'Gemma-3n-E2B-it','gemma-e4b':'Gemma-3n-E4B-it','minicpm-v45':'MiniCPM-V-4.5','internvl35-8b':'InternVL3.5-8B','internvl35-14b':'InternVL3.5-14B'}
def main():
    manifest=json.loads((ROOT/'benchmarks/scienceqa-test-100-manifest.json').read_text())
    expected={row['id']:row['expected'] for row in manifest['rows']}
    models=[]
    for key,name in NAMES.items():
        report=json.loads((ROOT/'evidence/scienceqa'/f'{key}.json').read_text())
        assert report['status']=='completed'
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
        models.append(dict(model_id=key,name=name,count=len(rows),native_correct=native,decision_correct=decision,agreements=agree,max_full_vocab_abs_diff=max(r['full_vocab_max_abs_diff'] for r in rows),changed_parameter_version_count=report['summary']['changed_parameter_version_count']))
    summary=dict(dataset=manifest['dataset'],suite_id=manifest['suite_id'],suite_sha256=manifest['suite_sha256'],models=models)
    (ROOT/'evidence/scienceqa/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    table='| Model | Native accuracy | Decision accuracy | Difference (pp) | Agreement | Max logit difference |\n|---|---:|---:|---:|---:|---:|\n'
    for r in models:
        n=r['count']
        table+=f"| {r['name']} | {100*r['native_correct']/n:.1f}% | {100*r['decision_correct']/n:.1f}% | {100*(r['decision_correct']-r['native_correct'])/n:+.1f} | {100*r['agreements']/n:.1f}% | {r['max_full_vocab_abs_diff']:g} |\n"
    (ROOT/'docs/scienceqa-results.md').write_text('# ScienceQA paired evaluation\n\n100 fixed image-bearing questions from the official test split, seed 42. Each backbone uses the same subset and paired prepared inputs.\n\n'+table+'\n[Method](scienceqa.md) · [Manifest](../benchmarks/scienceqa-test-100-manifest.json) · [Raw records](../evidence/scienceqa/)\n')
    print(table)
if __name__=='__main__': main()
