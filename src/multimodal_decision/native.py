"""Native visual embedding adapters; decisions use first-token candidate logits."""
import copy
import os
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BatchFeature
from .model import DecisionModel
from .contracts import LABELS

class NativeDecisionModel(DecisionModel):
    def __init__(self, model_dir, family):
        self.model_dir = Path(model_dir)
        self.family = family
        self.adapter = None
        self.image_roots = [Path(x) for x in os.environ.get('DECISION_IMAGE_ROOTS', '').split(':') if x]
        kw = dict(trust_remote_code=True, local_files_only=True)
        if family == 'minicpm':
            self.processor = AutoProcessor.from_pretrained(self.model_dir, **kw)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModel.from_pretrained(self.model_dir, torch_dtype=torch.bfloat16,
                attn_implementation='eager', **kw).to('cuda').eval()
            self.model.llm.set_attn_implementation('sdpa')
        elif family == 'internvl':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False, **kw)
            self.model = AutoModel.from_pretrained(self.model_dir, torch_dtype=torch.bfloat16,
                use_flash_attn=False, **kw).to('cuda').eval()
            self.model.language_model.set_attn_implementation('sdpa')
            self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        else:
            raise ValueError('Unknown native family')
        self.label_ids = []
        for label in LABELS:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError('Decision label must be a single token: ' + label)
            self.label_ids.append(ids[0])

    def prepare(self, request):
        content, images, provenance, keys = self.context(request)
        marker = '(<image>./</image>)' if self.family == 'minicpm' else '<image>'
        text = '\n'.join(marker if x['type'] == 'image' else x['text'] for x in content)
        if self.family == 'minicpm':
            text = self.tokenizer.apply_chat_template([dict(role='user', content=text)],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.processor([text], [images],
                return_tensors='pt', max_length=None, max_slice_nums=4) if images else BatchFeature(dict(self.tokenizer([text],return_tensors='pt')))
            inputs.pop('image_sizes', None)
            if not images:
                inputs['pixel_values'] = [[]]
                inputs['tgt_sizes'] = [[]]
                inputs['image_bound'] = [torch.empty((0,2),dtype=torch.long)]
            inputs = inputs.to('cuda')
        else:
            conversation = copy.deepcopy(self.model.conv_template)
            conversation.system_message = self.model.system_message
            conversation.append_message(conversation.roles[0], text)
            conversation.append_message(conversation.roles[1], None)
            text = conversation.get_prompt()
            pixels, counts = [], []
            for image in images:
                tiles = image_tiles(image, self.model.config.force_image_size or self.model.config.vision_config.image_size)
                counts.append(len(tiles)); pixels.extend(tiles)
            for count in counts:
                text = text.replace('<image>', '<img>' + '<IMG_CONTEXT>' * (self.model.num_image_token * count) + '</img>', 1)
            inputs = dict(self.tokenizer(text, return_tensors='pt'))
            inputs = {k:v.to('cuda') for k,v in inputs.items()}
            if pixels:
                inputs['pixel_values'] = torch.stack(pixels).to('cuda', dtype=torch.bfloat16)
        if inputs['input_ids'].shape[-1] > 8192:
            raise ValueError('Total input exceeds 8192-token control budget')
        return inputs, keys, provenance

    def visual_tokens(self, inputs):
        if self.family == 'minicpm':
            return sum(int(b-a) for a,b in inputs['image_bound'][0])
        return int((inputs['input_ids'] == self.model.img_context_token_id).sum())

    def logits(self, inputs):
        if self.family == 'minicpm':
            embeddings, _ = self.model.get_vllm_embedding(inputs)
            language = self.model.llm
        else:
            language = self.model.language_model
            embeddings = language.get_input_embeddings()(inputs['input_ids']).clone()
            pixels = inputs.get('pixel_values')
            if pixels is not None:
                features = self.model.extract_feature(pixels)
                selected = inputs['input_ids'] == self.model.img_context_token_id
                if int(selected.sum()) != features.shape[0] * features.shape[1]:
                    raise ValueError('Visual token count does not match image features')
                embeddings[selected] = features.reshape(-1, embeddings.shape[-1]).to(embeddings.dtype)
        return language(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'],
            use_cache=True, logits_to_keep=1).logits[0,-1].float()

    def native_first_step(self, inputs):
        kw = dict(max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True)
        if self.family == 'minicpm':
            output = self.model.generate(**inputs, tokenizer=self.tokenizer, decode_text=False, **kw)
        else:
            output = self.model.generate(**inputs, **kw)
        return output.scores[0][0].float()


def image_tiles(image, size, max_tiles=4):
    """InternVL's dynamic aspect-ratio tiling, with a bounded four-tile profile."""
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
    image = image.convert('RGB')
    w,h = image.size
    ratios = sorted({(i,j) for n in range(1,max_tiles+1) for i in range(1,n+1)
                     for j in range(1,n+1) if 1 <= i*j <= max_tiles}, key=lambda x:x[0]*x[1])
    best, difference = (1,1), float('inf')
    for ratio in ratios:
        delta = abs(w/h - ratio[0]/ratio[1])
        if delta < difference or (delta == difference and w*h > .5*size*size*ratio[0]*ratio[1]):
            best, difference = ratio, delta
    resized = image.resize((size*best[0],size*best[1]), Image.Resampling.BICUBIC)
    crops = [resized.crop((i%best[0]*size,i//best[0]*size,(i%best[0]+1)*size,(i//best[0]+1)*size))
             for i in range(best[0]*best[1])]
    if len(crops)>1:
        crops.append(image.resize((size,size), Image.Resampling.BICUBIC))
    transform = Compose([Resize((size,size), interpolation=InterpolationMode.BICUBIC), ToTensor(),
        Normalize((.485,.456,.406),(.229,.224,.225))])
    return [transform(crop) for crop in crops]
