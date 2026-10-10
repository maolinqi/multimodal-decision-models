"""One model resident at a time; existing Qwen service stays independent."""
import asyncio,gc,json,time
from pathlib import Path
import torch
from fastapi import FastAPI,HTTPException,Request
from .registry import MODELS as REGISTERED_MODELS, downloaded, create_model
# Qwen3.5 is served by qwen35_service in its separate runtime.
MODELS={k:v for k,v in REGISTERED_MODELS.items() if v[2]!='qwen35'}
ROOT=Path(__file__).resolve().parent

app=FastAPI(); lock=asyncio.Lock(); actor=None; active=None

@app.get('/health')
def health():
 return dict(ready=True,active_model=active,busy=lock.locked(),models={k:{'download_complete':downloaded(k),'loaded':active==k} for k in MODELS},flight_interface='proposal_only',real_flight_verified=False)

def release():
 global actor,active
 actor=None;active=None;gc.collect();torch.cuda.empty_cache()

def infer(payload,key):
 global actor,active
 begin=time.perf_counter();load_ms=0.
 if active!=key:
  release()
  free,_=torch.cuda.mem_get_info()
  minimum=MODELS[key][3]*1024**3
  if free<minimum:raise ValueError('显存不足以加载所选模型，请等待其他任务释放显存。')
  load_begin=time.perf_counter()
  actor=create_model(key);active=key
  load_ms=(time.perf_counter()-load_begin)*1000
 result=actor.velocity4(payload) if payload['type']=='velocity4' else actor.decide(payload)
 result.update(model_id=key,returned_at_ms=time.time()*1000,training_applied=bool(actor.adapter),model_load_ms=load_ms,request_latency_ms=(time.perf_counter()-begin)*1000,fresh_per_forward_cache=True,decoding="restricted_next_token_logits")
 return result

@app.post('/decide')
async def decide(request:Request):
 body=await request.body()
 if len(body)>24*1024*1024:raise HTTPException(413,'请求超过24MB')
 try:
  payload=json.loads(body)
  if not isinstance(payload,dict):raise ValueError('Request must be an object')
  key=payload.get('model_id',next(iter(MODELS)))
  if key not in MODELS:raise ValueError('Unknown model')
  if payload.get('type') not in {'choice','score','noul','velocity4'}:raise ValueError('Unknown decision type')
 except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc))
 if not downloaded(key):raise HTTPException(503,'所选模型仍在下载或下载失败，请查看模型状态。')
 if lock.locked():raise HTTPException(429,'模型正在推理或切换模型，请稍后再试。')
 async with lock:
  try:return await asyncio.to_thread(infer,payload,key)
  except (ValueError,KeyError,TypeError) as exc:raise HTTPException(422,str(exc))
  except torch.cuda.OutOfMemoryError:
   release();raise HTTPException(503,'显存不足，本次模型已卸载，请减少输入。')

@app.post('/unload')
async def unload():
 if lock.locked():raise HTTPException(409,'模型正在使用中')
 async with lock:await asyncio.to_thread(release)
 return {'unloaded':True}


@app.get('/downloads')
def downloads():
 return {'models':[{'name':v[0], 'complete':downloaded(k), 'state':'complete' if downloaded(k) else 'pending'} for k,v in MODELS.items()], 'all_complete':all(downloaded(k) for k in MODELS)}
