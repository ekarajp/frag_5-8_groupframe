"""Local-only dashboard. Opening it never starts production analyses."""
import argparse
import atexit
import json
import mimetypes
import os
import secrets
import signal
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read, write
from engine import Engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8788)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--state')
    parser.add_argument('--bind',choices=('127.0.0.1','0.0.0.0'),default='127.0.0.1',
                        help='0.0.0.0 only for a container mapped to host loopback')
    args = parser.parse_args()
    state = Path(args.state or ROOT/'state').resolve()
    state.mkdir(parents=True,exist_ok=True)
    # OS lock released on crash, unlike a stale heartbeat lock file.
    lock = (state/'server.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        if not args.no_browser:
            webbrowser.open(f'http://127.0.0.1:{args.port}')
        print('This portable campaign already has a dashboard/controller.')
        return
    engine = Engine(state)
    atexit.register(engine.close)
    token = secrets.token_urlsafe(32)
    export_lock = threading.Lock()
    export_status = {'state':'idle'}
    def export():
        with export_lock:
            export_status.update(state='working')
            try:
                engine.pool.submit(engine.finalize).result()
                export_status.update(state=engine.final_export_state)
            except Exception as e:
                export_status.update(state='error',error=str(e))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def send(self,code,data,ctype='application/json; charset=utf-8'):
            if isinstance(data,(dict,list)):
                data = json.dumps(data,ensure_ascii=True,allow_nan=False).encode()
            if isinstance(data,str):
                data = data.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type',ctype)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(data)
        def safe_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{args.port}'
        def do_GET(self):
            if not self.safe_host():
                return self.send(403,{'error':'Local dashboard Host required'})
            path = urlparse(self.path).path
            try:
                if path == '/api/status':
                    return self.send(200,{**engine.snapshot(),'token':token,'export':export_status})
                if path == '/api/reports':
                    return self.send(200,[{'name':p.parent.name,'path':'/results/'+p.relative_to(engine.run_dir).as_posix()}
                        for p in sorted((engine.run_dir/'reports').glob('*/index.html'))])
                if path.startswith('/results/'):
                    p = (engine.run_dir / unquote(path[len('/results/'):])).resolve()
                    if not p.is_relative_to(engine.run_dir) or not p.is_file():
                        return self.send(404,{'error':'File not found'})
                    return self.send(200,p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
                if path.startswith('/reference/'):
                    p=(ROOT/'assets'/unquote(path[len('/reference/'):])).resolve()
                    if not p.is_relative_to(ROOT/'assets') or not p.is_file():
                        return self.send(404,{'error':'Reference not found'})
                    return self.send(200,p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
                if path == '/':
                    return self.send(200,(ROOT/'app/dashboard.html').read_bytes(),'text/html; charset=utf-8')
                self.send(404,{'error':'Not found'})
            except Exception as e:
                self.send(500,{'error':str(e)})
        def do_POST(self):
            if not self.safe_host() or self.headers.get('X-Runner-Token') != token:
                return self.send(403,{'error':'Invalid local request token'})
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{args.port}':
                return self.send(403,{'error':'Cross-origin request refused'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if length > 4096:
                    raise ValueError('Request too large')
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/api/start':
                    engine.start(body.get('workers',7))
                elif self.path == '/api/stop':
                    engine.stop()
                elif self.path == '/api/workers':
                    engine.set_workers(body['workers'])
                elif self.path == '/api/export':
                    if engine.processes or engine.active:
                        raise ValueError('Stop and drain before creating a consistent full research inventory')
                    if export_status['state'] != 'working':
                        engine.final_export_state='queued'
                        threading.Thread(target=export,daemon=True).start()
                else:
                    return self.send(404,{'error':'Unknown action'})
                self.send(200,{'ok':True})
            except (ValueError,KeyError) as e:
                self.send(400,{'error':str(e)})
            except Exception as e:
                self.send(500,{'error':str(e)})
    httpd = ThreadingHTTPServer((args.bind,args.port),Handler)
    write(state/'dashboard.json',{'url':f'http://127.0.0.1:{args.port}','pid':os.getpid(),
                                'production_autostart':False})
    if not args.no_browser:
        webbrowser.open(f'http://127.0.0.1:{args.port}')
    print(f'Dashboard ready: http://127.0.0.1:{args.port} — waiting for Start',flush=True)
    try:
        httpd.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        httpd.server_close()

if __name__ == '__main__':
    main()
