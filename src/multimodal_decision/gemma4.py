"""Frozen Gemma 4 adapter, native vision and instruction chat template.
Requires the separate Transformers 5.19 runtime; no main-environment upgrade.
"""
import os
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor
from .model import DecisionModel
from .contracts import LABELS

class Gemma4DecisionModel(DecisionModel):
    def __init__(self,model_dir):
        import transformers
        if transformers.__version__!='5.19.0':
            raise ValueError('Gemma 4 requires the isolated .venv-gemma4 runtime (Transformers 5.19.0)')
        self.model_dir=Path(model_dir);self.adapter=None;self.family='gemma4'
        self.image_roots=[Path(x) for x in os.environ.get('DECISION_IMAGE_ROOTS','').split(':') if x]
        torch.set_num_threads(8)
        self.processor=AutoProcessor.from_pretrained(self.model_dir,local_files_only=True)
        self.processor.image_processor.max_soft_tokens=280
        self.processor.image_seq_length=280
        self.tokenizer=self.processor.tokenizer
        self.model=AutoModelForImageTextToText.from_pretrained(self.model_dir,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
        self.model.requires_grad_(False)
        self.label_ids=[]
        for label in LABELS:
            ids=self.tokenizer.encode(label,add_special_tokens=False)
            if len(ids)!=1:raise ValueError('Candidate label is not a single token: '+label)
            self.label_ids.append(ids[0])

    def prepare(self,request):
        content,images,provenance,keys=self.context(request)
        messages=[{'role':'user','content':content}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=self.tokenizer.encode(text,add_special_tokens=False)
        for i in range(len(keys)):
            if self.tokenizer.encode(text+LABELS[i],add_special_tokens=False)!=prefix+[self.label_ids[i]]:
                raise ValueError('Candidate token boundary changed')
        inputs=self.processor(text=[text],images=images or None,return_tensors='pt')
        if inputs['input_ids'].shape[-1]>8192:raise ValueError('Total input exceeds 8192-token control budget')
        return inputs.to(device='cuda',dtype=torch.bfloat16),keys,provenance

    def visual_tokens(self,inputs):
        return int((inputs['input_ids']==self.model.config.image_token_id).sum())

    def logits(self,inputs):
        return self.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()

    def native_first_step(self,inputs):
        output=self.model.generate(**inputs,max_new_tokens=1,do_sample=False,use_cache=True,return_dict_in_generate=True,output_logits=True)
        return output.logits[0][0].float()
