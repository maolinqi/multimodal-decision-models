"""Independent UI gateway. Calls the existing loopback decision service only."""
import asyncio
import base64
import io
import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from collections import defaultdict, deque
import httpx
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT=Path(__file__).resolve().parents[2]
MODEL_URL=os.environ.get('DECISION_URL','http://127.0.0.1:8457')
MODEL_URLS={key:MODEL_URL for key in ['gemma-e4b','gemma-e2b','minicpm-v45','internvl35-8b','internvl35-14b']}
MODEL_URLS['gemma4-a4b']=os.environ.get('GEMMA4_DECISION_URL','http://127.0.0.1:8461')
MODEL_URLS['qwen35-2b']=os.environ.get('QWEN35_DECISION_URL','http://127.0.0.1:8460')
if os.environ.get('QWEN_DECISION_URL'):MODEL_URLS['qwen4b']=os.environ['QWEN_DECISION_URL']
MODEL_NAMES={'gemma4-a4b':'Gemma-4-26B-A4B-it','qwen35-2b':'Qwen3.5-2B-Base','qwen4b':'Qwen3-VL-4B-Instruct','gemma-e4b':'Gemma-3n-E4B-it','gemma-e2b':'Gemma-3n-E2B-it','minicpm-v45':'MiniCPM-V-4.5','internvl35-8b':'InternVL3.5-8B','internvl35-14b':'InternVL3.5-14B'}
STATE_KEYS={'goal_error_vehicle_m','velocity_vehicle_mps','height_m','attitude_quaternion_xyzw',
    'angular_velocity_body_radps','battery_fraction','localization_valid','localization_covariance',
    'sensor_coverage','depth_sectors_m','mission','previous_velocity4_command','measured_history',
    'observation_kind','state_timestamp_ms'}
app=FastAPI(docs_url=None,redoc_url=None)
busy=asyncio.Lock()
media_slots=asyncio.Semaphore(2)
requests_by_ip=defaultdict(deque)

@app.middleware('http')
async def headers_and_limits(request,call_next):
    if int(request.headers.get('content-length','0') or 0)>64*1024*1024:
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'上传超过64MB，请缩短视频或压缩文件。'},status_code=413)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    response.headers['Cache-Control']='no-store' if request.url.path.startswith('/api/') else 'no-cache'
    return response

@app.get('/api/health')
async def health(model_id:str='gemma-e4b'):
    if model_id not in MODEL_URLS:raise HTTPException(422,'未知模型')
    try:
        async with httpx.AsyncClient(trust_env=False,timeout=3) as c:
            r=await c.get(MODEL_URLS[model_id]+'/health');r.raise_for_status();data=r.json()
        ready=bool(data.get('ready')) if model_id=='qwen4b' else data.get('models',{}).get(model_id,{}).get('download_complete',False)
        return dict(ui_ready=True,model_ready=ready,busy=busy.locked() or data.get('busy',False),model=MODEL_NAMES[model_id],model_id=model_id,loaded=data.get('active_model')==model_id if model_id!='qwen4b' else ready)
    except httpx.HTTPError:
        return dict(ui_ready=True,model_ready=False,busy=busy.locked(),model=MODEL_NAMES[model_id],model_id=model_id)

@app.get('/api/models')
async def models():
    return {'models':[await health(key) for key in MODEL_URLS]}

@app.get('/api/downloads')
async def downloads():
    try:
        async with httpx.AsyncClient(trust_env=False,timeout=10) as client:
            r=await client.get(MODEL_URL+'/downloads');r.raise_for_status();return r.json()
    except httpx.HTTPError:raise HTTPException(503,'暂时无法读取下载状态')

@app.get('/api/capabilities')
def capabilities():
    return dict(modalities=['text','rgb_image','video_sampled_rgb_frames'],min_options=2,max_options=26,
        max_frames=8,max_image_bytes=8*1024*1024,max_source_pixels=16000000,
        max_video_bytes=50*1024*1024,max_question_chars=8000,max_state_chars=16000,
        state_fields=sorted(STATE_KEYS),image_long_edge=1024,visual_token_budget_per_image=None,total_input_token_budget=8192,
        calibrated=False,audio=False,video_native_temporal_tokens=False)

def validate_image(raw):
    if len(raw)>8*1024*1024:raise ValueError('单张图片不能超过8MB。')
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width*im.height>16000000:raise ValueError('图片超过1600万像素，请先缩小图片。')
            if im.format not in {'JPEG','PNG','WEBP','BMP'}:raise ValueError('请上传JPEG、PNG、WEBP或BMP图片。')
            im.load();width,height=im.size
    except (UnidentifiedImageError,OSError):raise ValueError('无法读取图片，请检查文件是否损坏。')
    return width,height

def frame(raw,timestamp,camera,name):
    w,h=validate_image(raw)
    with Image.open(io.BytesIO(raw)) as image:mime=Image.MIME.get(image.format,'image/jpeg')
    return dict(base64=base64.b64encode(raw).decode(),timestamp_ms=timestamp,camera_id=camera,
        modality='rgb',name=name,width=w,height=h,mime=mime)

def extract_video(raw,start,end,count):
    with tempfile.TemporaryDirectory(prefix='decision-ui-video-') as temp:
        path=Path(temp)/'upload.video';path.write_bytes(raw)
        try:
            probe=subprocess.run(['ffprobe','-v','error','-protocol_whitelist','file,pipe',
                '-show_entries','format=duration','-of','json',str(path)],capture_output=True,timeout=20,check=True)
            duration=float(json.loads(probe.stdout)['format']['duration'])
            stop=duration if end is None else min(end,duration)
            if not math.isfinite(duration) or not 0<=start<stop:raise ValueError('视频时间范围无效。')
            times=[start+(stop-start)*(i+.5)/count for i in range(count)]
            frames=[];camera='video-'+uuid.uuid4().hex[:8]
            for i,ts in enumerate(times):
                image=subprocess.run(['ffmpeg','-v','error','-protocol_whitelist','file,pipe','-ss',str(ts),
                    '-i',str(path),'-frames:v','1','-vf','scale=1024:1024:force_original_aspect_ratio=decrease',
                    '-f','image2pipe','-vcodec','mjpeg','-q:v','2','pipe:1'],capture_output=True,check=True,timeout=30).stdout
                frames.append(frame(image,round(ts*1000,2),camera,f'帧 {i+1} · {ts:.2f}s'))
            return dict(frames=frames,source='video',duration_s=duration,start_s=start,end_s=stop,
                adaptation='ordered_rgb_frames',notice='视频以带时间戳的RGB帧输入；最多8帧，不含音轨。')
        except (subprocess.SubprocessError,KeyError,TypeError,json.JSONDecodeError):
            raise ValueError('视频无法解码，请使用常见MP4/MOV/WebM文件并缩短片段。')

@app.post('/api/media')
async def media(file:UploadFile=File(...),kind:str=Form('image'),start:float=Form(0),
                end:str=Form(''),count:int=Form(4)):
    if kind not in {'image','video'}:raise HTTPException(422,'仅支持图片或视频。')
    maximum=(8 if kind=='image' else 50)*1024*1024
    raw=await file.read(maximum+1);await file.close()
    if len(raw)>maximum:raise HTTPException(413,f'文件超过{maximum//1024//1024}MB。')
    try:
        if kind=='image':return dict(frames=[frame(raw,0,'image',file.filename or '图片')],source='image')
        if not 1<=count<=8 or not math.isfinite(start):raise ValueError('抽帧数量应为1–8，时间应为有限数值。')
        stop=float(end) if end.strip() else None
        if stop is not None and not math.isfinite(stop):raise ValueError('结束时间无效。')
        async with media_slots:return await asyncio.to_thread(extract_video,raw,start,stop,count)
    except ValueError as exc:raise HTTPException(422,str(exc))

def build_request(body):
    question=body.get('question','').strip()
    if not question or len(question)>8000:raise ValueError('请填写问题，最多8000字。')
    state=body.get('state_json',{})
    if not isinstance(state,dict):raise ValueError('结构化状态必须是JSON对象。')
    if set(state)-STATE_KEYS:raise ValueError('不支持的状态字段：'+','.join(sorted(set(state)-STATE_KEYS)))
    state=dict(state);text=body.get('state_text','').strip()
    if text:
        if 'mission' in state:state['mission']={'description':text,'structured':state['mission']}
        else:state['mission']=text
    if len(json.dumps(state,ensure_ascii=False,allow_nan=False))>16000:raise ValueError('状态内容最多16000字。')
    options=body.get('options',[])
    if not isinstance(options,list) or not 2<=len(options)<=26:raise ValueError('请提供2–26个选项。')
    mapping={}
    for item in options:
        key=item['id'];description=item['text'].strip()
        if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',key) or key in mapping:raise ValueError('选项ID必须唯一。')
        if not description or len(description)>4000:raise ValueError('每个选项需填写内容，最多4000字。')
        mapping[key]=description
    images=body.get('images',[])
    if not isinstance(images,list) or len(images)>8:raise ValueError('图片和视频帧合计最多8张。')
    accepted=[]
    for item in images:
        raw=base64.b64decode(item['base64'],validate=True);validate_image(raw)
        timestamp=float(item.get('timestamp_ms',0))
        if not math.isfinite(timestamp):raise ValueError('帧时间无效。')
        accepted.append(dict(base64=item['base64'],timestamp_ms=timestamp,
            camera_id=str(item.get('camera_id','image'))[:100],modality='rgb'))
    payload=dict(type='choice',question=question,state=state,options=mapping,images=accepted)
    if len(json.dumps(payload,ensure_ascii=False).encode())>24*1024*1024:raise ValueError('图像总量超过接口限制，请压缩图片或减少帧数。')
    return payload

@app.post('/api/decide')
async def decide(request:Request):
    raw=b''
    async for chunk in request.stream():
        raw+=chunk
        if len(raw)>24*1024*1024:raise HTTPException(413,'请求过大，请减少图片或压缩文件。')
    try:
        body=json.loads(raw); model_id=body.get('model_id','gemma-e4b')
        if model_id not in MODEL_URLS:raise ValueError('未知模型')
        payload=build_request(body)
        if model_id!='qwen4b':payload['model_id']=model_id
    except (ValueError,TypeError,KeyError,AttributeError) as exc:raise HTTPException(422,str(exc))
    ip=request.headers.get('cf-connecting-ip') or (request.client.host if request.client else 'local')
    history=requests_by_ip[ip];now=time.monotonic()
    while history and now-history[0]>60:history.popleft()
    if len(history)>=12:raise HTTPException(429,'请求较频繁，请稍后再试。')
    if busy.locked():raise HTTPException(429,'模型正在处理其他请求，请稍后再试。')
    history.append(now)
    async with busy:
        try:
            async with httpx.AsyncClient(trust_env=False,timeout=240) as client:
                # A single GPU can serve all seven backbones sequentially.
                # Unload only this console's other managed backends, never external Qwen.
                target=MODEL_URLS[model_id]
                managed={url for key,url in MODEL_URLS.items() if key!='qwen4b'}
                if model_id!='qwen4b':
                    for url in sorted(managed-{target}):
                        try:
                            check=await client.get(url+'/health',timeout=3)
                        except httpx.ConnectError:
                            continue
                        except httpx.TimeoutException:
                            raise HTTPException(503,'其他模型服务响应超时，暂时无法安全切换。')
                        if check.status_code!=200:
                            raise HTTPException(503,'其他模型服务状态异常，暂时无法切换。')
                        if check.json().get('active_model'):
                            released=await client.post(url+'/unload')
                            if released.status_code!=200:
                                raise HTTPException(409,'其他模型正在使用中，请稍后切换。')
                r=await client.post(target+'/decide',json=payload)
        except httpx.ConnectError:raise HTTPException(503,'模型服务尚未连接，请通过桌面启动文件启动。')
        except httpx.TimeoutException:raise HTTPException(504,'本次推理超时，请减少图片或缩短文字。')
        if r.status_code!=200:
            try:detail=r.json().get('detail','模型请求失败。')
            except ValueError:detail='模型请求失败。'
            raise HTTPException(r.status_code,detail)
        data=r.json();distribution=data.get('distribution',{})
        if set(distribution)!=set(payload['options']) or not all(isinstance(p,(int,float)) and math.isfinite(p) and p>=0 for p in distribution.values()) or abs(sum(distribution.values())-1)>1e-4:
            raise HTTPException(502,'模型返回了无效概率分布。')
        winner=max(distribution,key=distribution.get)
        if data.get('answer')!=winner:raise HTTPException(502,'模型选择与概率分布不一致。')
        return dict(answer=winner,answer_text=payload['options'][winner],distribution=distribution,
            options=payload['options'],latency_ms=data.get('request_latency_ms',data['latency_ms']),inference_latency_ms=data['latency_ms'],model_load_ms=data.get('model_load_ms',0),input_tokens=data['input_tokens'],
            label_mass=data.get('label_mass'),forward_passes=data.get('forward_passes'),base_model=data.get('base_model'),image_provenance=data.get('images'),
            visual_tokens=data['visual_tokens'],image_count=len(payload['images']),calibrated=False,
            model=MODEL_NAMES[model_id],model_id=model_id,adapter_loaded=bool(data.get('adapter')),returned_at_ms=data['returned_at_ms'])

@app.get('/')
def index():return FileResponse(ROOT/'web/index.html')
app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
