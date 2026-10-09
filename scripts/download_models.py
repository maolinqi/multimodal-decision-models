"""Download official model snapshots directly to the configured model root."""
import argparse, os, json
from pathlib import Path
from huggingface_hub import snapshot_download
MODELS = {
 'gemma-e2b': ('google/gemma-3n-E2B-it','gemma-3n-E2B-it'),
 'gemma-e4b': ('google/gemma-3n-E4B-it','gemma-3n-E4B-it'),
 'minicpm-v45': ('openbmb/MiniCPM-V-4_5','MiniCPM-V-4_5'),
 'internvl35-8b': ('OpenGVLab/InternVL3_5-8B','InternVL3_5-8B'),
 'internvl35-14b': ('OpenGVLab/InternVL3_5-14B','InternVL3_5-14B'),
}
p=argparse.ArgumentParser();p.add_argument('models',nargs='+',choices=list(MODELS))
p.add_argument('--root',default=os.environ.get('MODEL_ROOT','models'));p.add_argument('--revision',default='main')
a=p.parse_args()
for key in a.models:
 repo,directory=MODELS[key];path=Path(a.root)/directory
 snapshot_download(repo,revision=a.revision,local_dir=path,
  ignore_patterns=['*.bin','*.pt','*.pth','*.onnx','*.gguf','*.msgpack','*.h5'])
 index=path/'model.safetensors.index.json'
 weights=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
 if not (path/'config.json').is_file() or not all((path/x).is_file() and (path/x).stat().st_size>0 for x in weights):
  raise RuntimeError('Incomplete snapshot: '+key)
 print(key, 'snapshot downloaded; GPU validation remains a separate step')
