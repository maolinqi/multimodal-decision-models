"""Verify downloaded original weights against independent pinned HF LFS hashes."""
import hashlib,json
from pathlib import Path
from download_models import MODELS, ROOT

if __name__=='__main__':
 expected=json.loads((ROOT/'reference/expected_weights.json').read_text());result={}
 for kind,(repo,rev,path) in MODELS.items():
  if expected[repo]['revision']!=rev:raise ValueError('Revision mismatch')
  for name,reference in expected[repo]['files'].items():
   p=Path(path)/name
   if p.stat().st_size!=reference['size']:raise ValueError('Weight size mismatch')
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
   actual=h.hexdigest()
   if actual!=reference['sha256']:raise ValueError('Weight checksum mismatch')
   result[f'{repo}/{name}']=dict(sha256=actual,bytes=p.stat().st_size,revision=rev)
   print(kind,name,'checksum verified',flush=True)
 (ROOT/'run/weight_integrity.json').write_text(json.dumps(dict(state='complete',files=result),indent=2))
