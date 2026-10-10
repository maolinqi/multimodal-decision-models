from .model import MODEL_ROOT
MODELS = {
    'qwen35-2b': ('Qwen3.5-2B-Base', 'Qwen/Qwen3.5-2B-Base', 'qwen35', 8),
    'gemma-e2b': ('gemma-3n-E2B-it', 'google/gemma-3n-E2B-it', 'gemma', 16),
    'gemma-e4b': ('gemma-3n-E4B-it', 'google/gemma-3n-E4B-it', 'gemma', 20),
    'minicpm-v45': ('MiniCPM-V-4_5', 'openbmb/MiniCPM-V-4_5', 'minicpm', 22),
    'internvl35-8b': ('InternVL3_5-8B', 'OpenGVLab/InternVL3_5-8B', 'internvl', 22),
    'internvl35-14b': ('InternVL3_5-14B', 'OpenGVLab/InternVL3_5-14B', 'internvl', 34),
}

def downloaded(key):
    import json
    path = MODEL_ROOT / MODELS[key][0]
    try:
        index = json.loads((path/'model.safetensors.index.json').read_text())
        return (path/'config.json').is_file() and all((path/name).is_file() for name in set(index['weight_map'].values()))
    except (OSError, ValueError, KeyError):
        return (path/'config.json').is_file() and (path/'model.safetensors').is_file()

def create_model(key):
    directory, _, family, _ = MODELS[key]
    if family == 'qwen35':
        from .qwen35 import Qwen35DecisionModel
        return Qwen35DecisionModel(MODEL_ROOT/directory)
    if family == 'gemma':
        from .model import DecisionModel
        return DecisionModel(model_dir=MODEL_ROOT/directory)
    from .native import NativeDecisionModel
    return NativeDecisionModel(MODEL_ROOT/directory, family)
