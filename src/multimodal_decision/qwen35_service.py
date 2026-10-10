"""One Qwen3.5 model service in its project-local isolated runtime."""
from . import service
from .registry import MODELS
service.MODELS={'qwen35-2b':MODELS['qwen35-2b']}
app=service.app
