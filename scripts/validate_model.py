"""Small real-GPU interface probes, including official first-generation-step parity."""
import argparse, base64, io, json, math, time
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from multimodal_decision.registry import create_model, MODELS
p=argparse.ArgumentParser();p.add_argument('model',choices=list(MODELS));p.add_argument('--out',required=True)
a=p.parse_args();print('LOADING',a.model,flush=True);actor=create_model(a.model);print('LOADED',a.model,flush=True)
results=[]
with torch.inference_mode():
 for side in ['left','right']:
  image=Image.new('RGB',(384,256),'white');draw=ImageDraw.Draw(image)
  x=70 if side=='left' else 314;draw.ellipse((x-32,96,x+32,160),fill='red')
  buf=io.BytesIO();image.save(buf,format='PNG')
  req=dict(type='choice',state={},question='红色圆形位于画面的左侧还是右侧？',options={'left':'左侧','right':'右侧'},
   images=[dict(base64=base64.b64encode(buf.getvalue()).decode(),timestamp_ms=0,camera_id='front')])
  print('POSITION',side,flush=True)
  result=actor.decide(req)
  assert result['visual_tokens']>0 and abs(sum(result['distribution'].values())-1)<1e-5
  inputs,keys,_=actor.prepare(req)
  direct=actor.logits(inputs)
  if hasattr(actor,'native_first_step'):native=actor.native_first_step(inputs)
  else:native=actor.model.generate(**inputs,max_new_tokens=1,do_sample=False,return_dict_in_generate=True,output_scores=True).scores[0][0].float()
  delta=float((direct[actor.label_ids[:len(keys)]]-native[actor.label_ids[:len(keys)]]).abs().max())
  assert delta<=1e-3, f'Native candidate logits mismatch: {delta}'
  print('PARITY',side,delta,flush=True)
  results.append(dict(expected=side,correct=result['answer']==side,result=result,native_candidate_max_abs_diff=delta))
 for kind,extra in [('choice',dict(options={'four':'4','five':'5','six':'6'})),('noul',{}),('score',dict(levels=['低','中','高']))]:
  print('TYPED',kind,flush=True)
  result=actor.decide(dict(type=kind,state={},images=[],question='2+3等于5吗？' if kind=='noul' else '2+3等于多少？',**extra))
  assert abs(sum(result['distribution'].values())-1)<1e-5
  results.append(dict(probe=kind,result=result))
 result=actor.velocity4(dict(state={'goal_error_vehicle_m':[2.,1.,.5],'velocity_vehicle_mps':[0.,0.,0.],'height_m':1.5},images=[]))
 assert result['forward_passes']==4 and result['executable'] is False
 results.append(dict(probe='velocity4',result=result))
report=dict(model_id=a.model,status='real_gpu_interface_validation_passed',results=results,
 synthetic_position_correct=sum(x.get('correct',False) for x in results),synthetic_position_count=2,
 training_applied=False,navigation_improvement_demonstrated=False,real_flight_verified=False,
 torch_version=torch.__version__,transformers_version=__import__('transformers').__version__)
# Public evidence does not expose deployment paths.
for row in results:
 result=row['result'];result['base_model']=MODELS[a.model][1]
 for component in result.get('components',[]):component['base_model']=MODELS[a.model][1]
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(dict(model_id=a.model,status=report['status'],position_correct=report['synthetic_position_correct']),ensure_ascii=False))
