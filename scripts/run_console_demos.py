"""Submit demo inputs only, preserving every response. Answers are checked separately."""
import argparse,base64,json,time
from pathlib import Path
import httpx
parser=argparse.ArgumentParser()
parser.add_argument('--model',default='minicpm-v45')
parser.add_argument('--url',default='http://127.0.0.1:8456')
parser.add_argument('--out',type=Path,default=Path('evidence/console-demos-rerun'))
parser.add_argument('demos',nargs='*',default=['civil-2023-73','civil-2022-74','civil-2023-74','civil-2019-73'])
a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[1]/'web/demos'
with httpx.Client(timeout=240,trust_env=False) as client:
 for key in a.demos:
  example=json.loads((root/(key+'.json')).read_text())
  frames=[dict(base64=base64.b64encode((root/im['file']).read_bytes()).decode(),
               camera_id='demo',timestamp_ms=im['timestamp_ms'],modality='rgb') for im in example['images']]
  payload=dict(model_id=a.model,question=example['question'],state_text=example['state_text'],
               options=[dict(id=chr(65+i),text=text) for i,text in enumerate(example['options'])],images=frames)
  begin=time.perf_counter();response=client.post(a.url+'/api/decide',json=payload)
  record=dict(input={**payload,'images':example['images']},result=response.json(),
              status_code=response.status_code,wall_ms=(time.perf_counter()-begin)*1000)
  (a.out/(key+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
  print(key,response.status_code,record['result'].get('answer'),flush=True)
  response.raise_for_status()
