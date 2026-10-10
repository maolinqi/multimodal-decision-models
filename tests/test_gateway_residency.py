"""A cross-family console switch must release other managed GPU residents."""
import asyncio
from unittest.mock import AsyncMock
import pytest
pytest.importorskip("fastapi", reason="Gateway integration checks require project runtime dependencies")
pytest.importorskip("httpx", reason="Gateway integration checks require project runtime dependencies")
pytest.importorskip("multipart", reason="Gateway integration checks require project runtime dependencies")
from fastapi import HTTPException
from starlette.requests import Request
from multimodal_decision import gateway
import json

class Response:
    def __init__(self,status,data): self.status_code,self.data=status,data
    def json(self): return self.data

class Backend:
    def __init__(self,release_status=200): self.calls=[];self.release_status=release_status
    async def __aenter__(self): return self
    async def __aexit__(self,*args): pass
    async def get(self,url,**kwargs):
        self.calls.append(('get',url))
        return Response(200,{'active_model':'gemma4-a4b' if '8461' in url else None})
    async def post(self,url,**kwargs):
        self.calls.append(('post',url))
        if url.endswith('/unload'): return Response(self.release_status,{'unloaded':True})
        return Response(200,{'answer':'yes','distribution':{'yes':1.,'no':0.},'latency_ms':5,
                             'input_tokens':10,'visual_tokens':0,'returned_at_ms':1})

def request():
    body=json.dumps({'model_id':'gemma-e4b','question':'Ready?',
        'options':[{'id':'yes','text':'yes'},{'id':'no','text':'no'}]}).encode()
    return Request({'type':'http','headers':[],'client':('test',1)},
        AsyncMock(return_value={'type':'http.request','body':body,'more_body':False}))

def prepare(monkeypatch,backend):
    monkeypatch.setattr(gateway.httpx,'AsyncClient',lambda **kwargs:backend)
    monkeypatch.setattr(gateway,'busy',asyncio.Lock())
    gateway.requests_by_ip.clear()

def test_switch_releases_before_inference(monkeypatch):
    backend=Backend();prepare(monkeypatch,backend)
    result=asyncio.run(gateway.decide(request()))
    assert result['answer']=='yes'
    release=('post','http://127.0.0.1:8461/unload')
    infer=('post',gateway.MODEL_URL+'/decide')
    assert backend.calls.index(release)<backend.calls.index(infer)

def test_busy_sibling_prevents_inference(monkeypatch):
    backend=Backend(409);prepare(monkeypatch,backend)
    with pytest.raises(HTTPException) as exc: asyncio.run(gateway.decide(request()))
    assert exc.value.status_code==409
    assert not any(url.endswith('/decide') for _,url in backend.calls)
