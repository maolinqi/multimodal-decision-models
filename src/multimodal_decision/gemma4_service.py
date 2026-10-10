"""Gemma 4 in its own isolated runtime and model service."""
from . import service
from .registry import MODELS
service.MODELS={"gemma4-a4b":MODELS["gemma4-a4b"]}
app=service.app
