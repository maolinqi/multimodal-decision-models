"""One serial lane of a two-GPU latency extension; pauses rather than skipping failures."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--lane', required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    lane = plan['lanes'][a.lane]
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / (a.lane + '.lock')).open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = root / (a.lane + '-status.json')
    raw = Path(plan['suite']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['suite_sha256'] or len(json.loads(raw)['rows']) != 100:
        raise ValueError('Fixed ScienceQA suite changed')
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((Path(plan['project_root']) / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed: ' + name)
    for item in lane['models']:
        out = root / item['id']
        out.mkdir(exist_ok=True)
        if (out / 'summary.json').exists() and json.loads((out / 'summary.json').read_text()).get('complete'):
            continue
        free = int(subprocess.check_output(['nvidia-smi', '-i', lane['gpu_uuid'],
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        required = item['minimum_gib'] + 4
        if free < required * 1024:
            save(status, dict(state='paused_memory', model=item['id'], free_mib=free, required_gib=required))
            return 75
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=lane['gpu_uuid'], MODEL_ROOT=plan['model_root'],
                   PYTHONPATH=str(Path(plan['project_root']) / 'src'), OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1')
        env.pop('GEMMA_ADAPTER', None)
        log = root / (item['id'] + '.log')
        command = [item['python'], '-u', str(Path(__file__).with_name('benchmark_latency_extension.py')),
                   '--model', item['id'], '--suite', plan['suite'], '--out', str(out),
                   '--repeats', '3', '--minimum-gib', str(required),
                   '--prompt-style', plan.get('prompt_style', 'neutral')]
        with log.open('a') as handle:
            child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            save(status, dict(state='running', model=item['id'], gpu_uuid=lane['gpu_uuid'],
                              controller_pid=os.getpid(), child_pid=child.pid))
            code = child.wait()
        (root / (item['id'] + '.exit')).write_text(str(code) + '\n')
        if code:
            tail = log.read_text()[-8000:]
            memory = 'out of memory' in tail.lower() or 'PAUSED_MEMORY' in tail
            state = 'paused_memory' if memory else 'paused_failure'
            save(out / 'status.json', dict(state=state, model=item['id'], exit_code=code, log=str(log)))
            save(status, dict(state=state, model=item['id'], exit_code=code))
            return 75 if memory else code
    save(status, dict(state='complete', models=[m['id'] for m in lane['models']]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
