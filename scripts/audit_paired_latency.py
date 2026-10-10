"""Recompute published paired latency/quality summaries from every raw record.
This checks completed groups only; it never reruns inference, drops caps or treats
repetitions as independent questions. Bootstrap intervals remain in source summaries.
"""
import argparse, json, math, statistics
from pathlib import Path

def audit(directory):
    summary=json.loads((directory/'summary.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text())
    rows=[json.loads(line) for line in (directory/'measurements.jsonl').read_text().splitlines()]
    n=protocol['n']; repeats=protocol['repeats']
    assert summary['complete'] and repeats==3
    assert len(rows)==len({r['key'] for r in rows})==n*(2*repeats+1)
    assert all('error' not in r for r in rows)
    groups={}
    for row in rows: groups.setdefault(row['id'],[]).append(row)
    assert len(groups)==n
    pairs=[]
    for ident, rs in groups.items():
        assert len({r['prompt_sha256'] for r in rs})==1
        assert len({r['input_tokens'] for r in rs})==1
        assert len({r['gold'] for r in rs})==1
        for mode in ['answer_only','decision']:
            subset=[r for r in rs if r['mode']==mode]
            assert len(subset)==repeats and {r['repeat'] for r in subset}==set(range(repeats))
        control=[r for r in rs if r['mode']=='first_step_control']
        assert len(control)==1 and control[0]['candidate_agreement']
        assert control[0]['finite_pattern_matches']
        assert control[0]['candidate_max_abs_diff']==control[0]['full_vocab_max_abs_diff']==0
        pairs.append(tuple(statistics.median(r['response_ms'] for r in rs if r['mode']==mode) for mode in ['answer_only','decision']))
    native=statistics.median(a for a,b in pairs); decision=statistics.median(b for a,b in pairs)
    measured=summary['paired']['answer_only']
    for key,value in [('native_median_ms',native),('decision_median_ms',decision),('difference_ms',native-decision),('speedup_ratio',native/decision),('reduction_percent',100*(1-decision/native))]:
        assert math.isclose(measured[key],value,rel_tol=1e-12,abs_tol=1e-9),(directory,key)
    assert measured['n_questions']==n and summary['retention']['parameter_versions_unchanged']
    for mode in ['answer_only','decision','first_step_control']:
        rs=[r for r in rows if r['mode']==mode]; m=summary['modes'][mode]
        assert m['n']==len(rs) and m['errors']==0
        assert m['correct']==sum(r['correct'] for r in rs)
        assert m['hit_token_cap']==sum(bool(r.get('hit_token_cap')) for r in rs)
        assert m['generated_tokens_median']==statistics.median(r['generated_tokens'] for r in rs)
    try: group=str(directory.resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError: group=directory.name
    return dict(group=group,questions=n,repetitions=repeats,raw_calls=len(rows),native_median_ms=native,decision_median_ms=decision,reduction_percent=100*(1-decision/native),native_cap_calls=summary['modes']['answer_only']['hit_token_cap'],same_prompt_per_question=True,raw_first_step_exact_agreements=n)

def main():
    p=argparse.ArgumentParser(); p.add_argument('directories',nargs='*',type=Path); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'evidence/frozen-backbone'
    directories=a.directories or sorted(x.parent for group in ['latency','latency-short-answer'] for x in (root/group).glob('*/summary.json'))
    assert directories, 'No completed public latency groups'
    results=[audit(x) for x in directories]
    print(json.dumps(dict(status='published_raw_latency_audit_passed',groups=results,scope='Raw record and published aggregate consistency; protocol hashes and first-step agreement do not establish universal capability retention'),indent=2))
if __name__=='__main__': main()
