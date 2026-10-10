"""Detached serial controller: pause immediately on insufficient GPU memory."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from evaluate_accuracy_matrix import records, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if [m['parameter_elements'] for m in plan['models']] != sorted(m['parameter_elements'] for m in plan['models']):
        raise ValueError('Models must be ordered by actual stored parameter count')
    rows = records(plan)
    for item in plan['models']:
        if not Path(item['python']).is_file():
            raise ValueError('Missing isolated runtime: ' + item['python'])
    if args.preflight:
        print(json.dumps(dict(models=len(plan['models']), questions_per_model=len(rows),
                              total_predictions=len(rows)*len(plan['models']), order=[m['id'] for m in plan['models']])))
        return 0
    status = root / 'status.json'
    for index, item in enumerate(plan['models']):
        model_status = root / item['id'] / 'status.json'
        if model_status.exists() and json.loads(model_status.read_text()).get('state') == 'complete':
            continue
        free = int(subprocess.check_output(['nvidia-smi', '-i', plan['gpu_uuid'],
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        required = (item['minimum_gib'] + plan['headroom_gib']) * 1024
        if free < required:
            save(status, dict(state='paused_memory', model=item['id'], next_index=index,
                              free_mib=free, required_mib=required, completed_models=index))
            print('Paused: insufficient GPU memory for ' + item['id'], flush=True)
            return 75
        save(status, dict(state='running', model=item['id'], next_index=index,
                          completed_models=index, pid=os.getpid(), gpu_uuid=plan['gpu_uuid']))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['gpu_uuid'], MODEL_ROOT=plan['model_root'],
                   PYTHONPATH=plan['source_root'], OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1')
        env.pop('GEMMA_ADAPTER', None)
        log = root / (item['id'] + '.log')
        with log.open('a') as handle:
            child = subprocess.Popen([item['python'], '-u', str(Path(__file__).with_name('evaluate_accuracy_matrix.py')),
                                      '--plan', str(args.plan.resolve()), '--model', item['id']],
                                     env=env, stdout=handle, stderr=subprocess.STDOUT)
            save(status, dict(state='running', model=item['id'], next_index=index,
                              completed_models=index, pid=os.getpid(), child_pid=child.pid, gpu_uuid=plan['gpu_uuid']))
            code = child.wait()
        (root / (item['id'] + '.exit')).write_text(str(code) + '\n')
        if code:
            save(status, dict(state='paused_memory' if code == 75 else 'paused_failure',
                              model=item['id'], next_index=index, exit_code=code, completed_models=index))
            return code
    summaries = [json.loads((root / m['id'] / 'summary.json').read_text()) for m in plan['models']]
    save(root / 'summary.json', dict(state='complete', models=summaries, total_predictions=len(rows)*len(summaries)))
    save(status, dict(state='complete', completed_models=len(plan['models'])))
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
        code = 1
    sys.exit(code)
