"""Check the project's started API processes without loading any model."""
import time, os
from pathlib import Path
from urllib.request import build_opener, ProxyHandler
opener=build_opener(ProxyHandler({}))
checks=[('backend',8457,'/health'),('gateway',8456,'/api/capabilities')]
for name,port in [('qwen35',8460),('gemma4',8461)]:
    if Path('run',name+'.pid').exists(): checks.append((name,port,'/health'))
for name,port,path in checks:
    deadline=time.monotonic()+45
    while True:
        pid=int(Path('run',name+'.pid').read_text())
        try: os.kill(pid,0)
        except ProcessLookupError:
            raise SystemExit(f'{name} exited; inspect logs/{name}.log (for example a port conflict)')
        try:
            with opener.open(f'http://127.0.0.1:{port}{path}',timeout=2) as response:
                if response.status==200: break
        except OSError: pass
        if time.monotonic()>deadline:
            raise SystemExit(f'{name} did not start; inspect logs/{name}.log and run ./stop_all.sh')
        time.sleep(.5)
    print(name+' ready')
