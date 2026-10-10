"""Download only the two original backbones, directly on the server."""
import json, os, time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get('BENCH_MODEL_ROOT', ROOT / 'models'))
MODELS = {
    'qwen': ('Qwen/Qwen3.5-2B-Base', 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c', str(MODEL_ROOT / 'Qwen3.5-2B-Base')),
    'gemma': ('google/gemma-4-26B-A4B-it', '4d7ae4984b7db7de8f8457170b3f1a419ee76d52', str(MODEL_ROOT / 'gemma-4-26B-A4B-it')),
}

def write_status(value):
    p = ROOT / 'run/download_status.json'
    p.parent.mkdir(exist_ok=True)
    q = p.with_suffix('.tmp'); q.write_text(json.dumps(value, indent=2)); q.replace(p)

if __name__ == '__main__':
    results = {}
    for name, (repo, revision, dest) in MODELS.items():
        write_status(dict(state='running', current=name, pid=os.getpid(), results=results))
        started = time.time()
        try:
            snapshot_download(repo, revision=revision, local_dir=dest, max_workers=4)
            files = sorted(str(p.relative_to(dest)) for p in Path(dest).rglob('*') if p.is_file() and '.cache' not in p.parts)
            results[name] = dict(state='complete', repo=repo, revision=revision, path=dest, files=files, elapsed_s=time.time()-started)
        except Exception as exc:
            results[name] = dict(state='failed', repo=repo, revision=revision, error=type(exc).__name__, elapsed_s=time.time()-started)
            print(name, type(exc).__name__, flush=True)
        write_status(dict(state='running', results=results))
    write_status(dict(state='complete' if all(x['state']=='complete' for x in results.values()) else 'incomplete', results=results))
    if any(x['state']!='complete' for x in results.values()): raise SystemExit(1)
