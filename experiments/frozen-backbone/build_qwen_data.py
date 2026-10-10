"""Reconstruct the author's shared-RNG Cauldron evaluation selection, no model training."""
import ast, hashlib, io, json, random, re, types
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
REVISION = '847a98a779b1652d65111daf20c972dfcd333605'
SOURCE = ROOT / 'reference/decider_vision_data.py'
author=types.SimpleNamespace()
tree=ast.parse(SOURCE.read_text())
nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in ('png','parse')) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SUBSETS','LETTER_RX') for t in n.targets))]
namespace={'io':io,'re':re}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
author=types.SimpleNamespace(**{k:namespace[k] for k in ('SUBSETS','png','parse')})

def sha(b): return hashlib.sha256(b).hexdigest()

if __name__ == '__main__':
    out = ROOT / 'data/qwen'; out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0); counts = {}; rows = []
    # Earlier subsets must consume the same shuffle RNG even though only two tasks are scored.
    for sub, (held, _) in author.SUBSETS.items():
        ds = load_dataset('HuggingFaceM4/the_cauldron', sub, revision=REVISION, split='train', streaming=True)
        kept = []; cap = 1500 if held else 6000
        target = sub in ('raven', 'visual7w')
        for source_row, r in enumerate(ds):
            if len(r['images']) != 1 or not r['texts']: continue
            t = r['texts'][0]; parsed = author.parse(sub, t['user'], t['assistant'])
            if parsed is None: continue
            record = dict(source_row=source_row)
            if target:
                ctx, question, options, gold = parsed
                im = r['images'][0]
                if max(im.size) > 768: im = im.copy(); im.thumbnail((768, 768))
                record.update(context='This is a visual question about the image.' + ('\n'+ctx if ctx else ''), question=question, options=options, gold=gold, image_bytes=author.png(im))
            kept.append(record)
            if len(kept) >= cap: break
        rng.shuffle(kept); n_eval = min(500, len(kept)//5)
        counts[sub] = dict(kept=len(kept), evaluation_pool=n_eval, excluded_from_task_training=held)
        print(sub, counts[sub], flush=True)
        if target:
            for i, r in enumerate(kept[:n_eval][:300]):
                b = r.pop('image_bytes'); path=out/f'{sub}-{i:03d}.png'; path.write_bytes(b)
                rows.append(dict(id=f'{sub}:{r["source_row"]}', task=sub, **r, image=str(path.relative_to(ROOT)), image_sha256=sha(b)))
        if sub == 'visual7w': break
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (out/'records.jsonl').write_text(payload)
    (out/'manifest.json').write_text(json.dumps(dict(dataset='HuggingFaceM4/the_cauldron', revision=REVISION, records_sha256=sha(payload.encode()), author_source_sha256=sha(SOURCE.read_bytes()), counts=counts, scored_counts={t:sum(r['task']==t for r in rows) for t in ('raven','visual7w')}, protocol='author default cap and shared RNG; first 300 evaluation records; original sample identity not independently verified'),indent=2))
    if any(sum(r['task']==t for r in rows)!=300 for t in ('raven','visual7w')): raise RuntimeError('Cannot reconstruct 300 records per task')
