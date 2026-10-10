"""Real isolated-service round trip, using an inherited ephemeral loopback socket."""
import base64,io,json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def main():
 sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(32);url='http://127.0.0.1:'+str(sock.getsockname()[1]);log=ROOT/'logs/qwen35-api-validation.log';log.parent.mkdir(exist_ok=True)
 with log.open('w') as f:
  proc=subprocess.Popen([sys.executable,'-m','uvicorn','multimodal_decision.qwen35_service:app','--fd',str(sock.fileno())],pass_fds=[sock.fileno()],stdout=f,stderr=f)
  results=[]
  def call(path,payload=None):
   body=json.dumps(payload).encode() if payload is not None else None
   req=urllib.request.Request(url+path,data=body,headers={'Content-Type':'application/json'})
   return json.load(urllib.request.urlopen(req,timeout=240))
  try:
   deadline=time.monotonic()+90
   while True:
    if proc.poll() is not None:raise RuntimeError('Isolated API exited during startup')
    try:h=call('/health');break
    except (OSError,ValueError):
     if time.monotonic()>deadline:raise RuntimeError('Isolated API startup timed out')
     time.sleep(.5)
   assert list(h['models'])==['qwen35-2b'] and h['active_model'] is None
   for side,x in [('left',70),('right',314)]:
    im=Image.new('RGB',(384,256),'white');ImageDraw.Draw(im).ellipse((x-32,96,x+32,160),fill='red');b=io.BytesIO();im.save(b,format='PNG')
    r=call('/decide',dict(type='choice',state={},question='红色圆形位于画面的左侧还是右侧？',options={'left':'左侧','right':'右侧'},images=[dict(base64=base64.b64encode(b.getvalue()).decode(),timestamp_ms=0,camera_id='front')]))
    assert r['model_id']=='qwen35-2b' and r['visual_tokens']>0 and not r['training_applied'] and r['forward_passes']==1 and abs(sum(r['distribution'].values())-1)<1e-5
    r['base_model']='Qwen/Qwen3.5-2B-Base';results.append(dict(expected=side,correct=r['answer']==side,result=r))
   assert call('/unload',{})['unloaded'] and call('/health')['active_model'] is None
   report=dict(status='isolated_api_round_trip_passed',model_id='qwen35-2b',synthetic_position_correct=sum(r['correct'] for r in results),synthetic_position_count=2,results=results,scope='Two synthetic image requests, default model routing, native visual path, distribution and unload; no general capability or flight claim')
   p=ROOT/'evidence/qwen35-api-validation.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report['status'],report['synthetic_position_correct'],flush=True)
  finally:
   sock.close();proc.terminate()
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
if __name__=='__main__':main()
