"""Exercise the seven installed image decision routes; this is not an accuracy benchmark."""
import argparse, base64, json, math, time, hashlib
from pathlib import Path
import httpx

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8456')
parser.add_argument('--out',type=Path,default=Path('run/installation-check.json'))
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
fixture=json.loads((root/'web/demos/space-greenhouse.json').read_text())
image=(root/'web/demos'/fixture['images'][0]['file']).read_bytes()
models=['qwen35-2b','gemma4-a4b','gemma-e2b','gemma-e4b','minicpm-v45','internvl35-8b','internvl35-14b']
report={'purpose':'Seven-model installation smoke check, not an accuracy benchmark',
        'fixture':'space-greenhouse','image_sha256':hashlib.sha256(image).hexdigest(),
        'results':[],'all_passed':False}
args.out.parent.mkdir(parents=True,exist_ok=True)
with httpx.Client(timeout=300,trust_env=False) as client:
 for model in models:
  payload={'model_id':model,'question':fixture['question'],'state_text':fixture['state_text'],
    'options':[{'id':chr(65+i),'text':text} for i,text in enumerate(fixture['options'])],
    'images':[{'base64':base64.b64encode(image).decode(),'camera_id':'demo','timestamp_ms':0,'modality':'rgb'}]}
  begin=time.perf_counter()
  try:
   response=client.post(args.url+'/api/decide',json=payload);response.raise_for_status()
   result=response.json();distribution=result['distribution'];values=list(distribution.values())
   expected={option['id'] for option in payload['options']}
   assert set(distribution)==expected and result['answer'] in expected
   assert all(math.isfinite(v) and 0<=v<=1 for v in values) and abs(sum(values)-1)<1e-5
   assert result['visual_tokens']>0
   residents=client.get(args.url+'/api/models');residents.raise_for_status()
   loaded=[m['model_id'] for m in residents.json()['models'] if m.get('loaded') and m['model_id']!='qwen4b']
   assert loaded==[model],loaded
   record={'model_id':model,'passed':True,'wall_ms':(time.perf_counter()-begin)*1000,
           'result':result,'resident_models':loaded}
  except Exception as error:
   record={'model_id':model,'passed':False,'error':str(error)}
  report['results'].append(record)
  args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  print(model,'PASS' if record['passed'] else 'FAIL',flush=True)
report['all_passed']=all(item['passed'] for item in report['results'])
args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
raise SystemExit(0 if report['all_passed'] else 1)
