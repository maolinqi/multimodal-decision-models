from pathlib import Path
import base64,io,json,hashlib
from PIL import Image,ImageDraw,ImageFont
r=Path(__file__).resolve().parents[1]
try: font=ImageFont.truetype('DejaVuSans.ttf',56)
except OSError: font=ImageFont.load_default(size=56)
def image(kind,value):
 im=Image.new('RGB',(384,256),'white');d=ImageDraw.Draw(im)
 if kind=='position':
  x=75 if value=='left' else 309;d.ellipse((x-32,96,x+32,160),fill='red')
 elif kind=='color':d.rectangle((100,60,284,196),fill=value)
 elif kind=='count':
  for i in range(value):
   x=70+80*i;d.ellipse((x-20,108,x+20,148),fill='blue')
 elif kind=='ocr':d.text((80,85),value,font=font,fill='black')
 elif kind=='spatial':
  y=55 if value=='above' else 201;d.ellipse((160,y-22,204,y+22),fill='red');d.rectangle((150,100,214,156),fill='blue')
 buf=io.BytesIO();im.save(buf,format='PNG');return base64.b64encode(buf.getvalue()).decode()
def frames(items):return [dict(base64=image(k,v),timestamp_ms=i*100,camera_id='front') for i,(k,v) in enumerate(items)]
rows=[]
def add(id,family,q,options,answer,imgs=(),state=None):rows.append(dict(id=id,family=family,expected=answer,request=dict(type='choice',question=q,options=options,state=state or {},images=frames(imgs))))
add('math_1','text_math','2+3等于多少？',{'four':'4','five':'5','six':'6'},'five')
add('math_2','text_math','9减4等于多少？',{'four':'4','five':'5','six':'6'},'five')
add('language_1','text_language','英文单词cat指哪种动物？',{'cat':'猫','dog':'狗','bird':'鸟'},'cat')
add('language_2','text_language','“北”的反方向是哪一方？',{'east':'东','south':'南','west':'西'},'south')
add('logic_1','text_logic','所有苹果都是水果，这个物体是苹果。它是水果吗？',{'yes':'是','no':'否'},'yes')
add('logic_2','text_logic','小明比小红高，小红比小李高。谁最高？',{'ming':'小明','hong':'小红','li':'小李'},'ming')
for color in ['red','green']:add('color_'+color,'visual_color','画面中大色块是什么颜色？',{'red':'红色','green':'绿色','blue':'蓝色'},color,[('color',color)])
for count in [2,3]:add('count_'+str(count),'visual_count','图中有几个蓝色圆形？',{'one':'1个','two':'2个','three':'3个','four':'4个'},'two' if count==2 else 'three',[('count',count)])
for text in ['CAT','DOG']:add('ocr_'+text.lower(),'visual_ocr','图片中的英文文字是什么？',{'cat':'CAT','dog':'DOG','car':'CAR'},text.lower(),[('ocr',text)])
for side in ['left','right']:add('position_'+side,'visual_position','红色圆形在画面的哪一侧？',{'left':'左侧','right':'右侧'},side,[('position',side)])
for side in ['above','below']:add('spatial_'+side,'visual_relation','红色圆形在蓝色方块的什么方向？',{'above':'上方','below':'下方'},side,[('spatial',side)])
add('multi_red_green','multi_image','第一帧色块红色，第二帧色块绿色吗？',{'yes':'是','no':'否'},'yes',[('color','red'),('color','green')])
add('multi_green_red','multi_image','第一帧色块红色，第二帧色块绿色吗？',{'yes':'是','no':'否'},'no',[('color','green'),('color','red')])
add('temporal_left_right','temporal','按时间顺序，红色圆形向什么方向移动？',{'left':'向左','right':'向右'},'right',[('position','left'),('position','right')])
add('temporal_right_left','temporal','按时间顺序，红色圆形向什么方向移动？',{'left':'向左','right':'向右'},'left',[('position','right'),('position','left')])
(r/'benchmarks').mkdir(exist_ok=True)
(r/'benchmarks/base_retention_v1.json').write_text(json.dumps(dict(suite_id='base_retention_v1',source='fixed_synthetic_and_text_probes',seed=None,rows=rows),ensure_ascii=False,indent=2))
print('Created',len(rows),'fixed probes')
