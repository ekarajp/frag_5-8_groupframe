"""Real HTTP dashboard startup and access checks; never sends Start."""
import json,socket,subprocess,sys,tempfile,time,unittest,urllib.error,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def wait_status(p,url):
    deadline=time.monotonic()+30
    while True:
        try:
            with urllib.request.urlopen(url+'/api/status',timeout=2) as response:return json.load(response)
        except (OSError,urllib.error.URLError):
            if p.poll() is not None or time.monotonic()>deadline:raise AssertionError('Dashboard failed to start')
            time.sleep(.1)
class DashboardTests(unittest.TestCase):
    def test_external_interpreter_startup_is_idle_and_start_requires_token(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        with tempfile.TemporaryDirectory() as folder:
            p=subprocess.Popen([sys.executable,'-B',str(ROOT/'run.py'),'--no-browser','--port',str(port),'--state',folder],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            url=f'http://127.0.0.1:{port}'
            try:
                status=wait_status(p,url)
                self.assertFalse(status['running']);self.assertEqual(status['active'],[])
                self.assertEqual(status['models_total'],2000);self.assertEqual(status['overall_percent'],0)
                request=urllib.request.Request(url+'/api/start',data=b'{}',headers={'Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(request,timeout=2)
                self.assertEqual(error.exception.code,403)
                with urllib.request.urlopen(url+'/api/status',timeout=2) as response:after=json.load(response)
                self.assertFalse(after['running']);self.assertEqual(after['active'],[])
                self.assertFalse(list(Path(folder).glob('run-*/jobs/*/job.json')))
            finally:
                p.terminate();p.wait(timeout=10)
    def test_allowed_host_header_grants_access_and_token_still_required(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        with tempfile.TemporaryDirectory() as folder:
            p=subprocess.Popen([sys.executable,'-B',str(ROOT/'run.py'),'--no-browser','--port',str(port),'--state',folder,
                                '--allowed-host',f'127.0.0.2:{port}'],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            url=f'http://127.0.0.1:{port}';allowed=f'127.0.0.2:{port}'
            try:
                status=wait_status(p,url)
                self.assertTrue(status['token'])
                request=urllib.request.Request(url+'/api/status',headers={'Host':allowed})
                with urllib.request.urlopen(request,timeout=2) as response:self.assertTrue(json.load(response)['token'])
                request=urllib.request.Request(url+'/api/status',headers={'Host':f'127.0.0.3:{port}'})
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(request,timeout=2)
                self.assertEqual(error.exception.code,403)
                request=urllib.request.Request(url+'/api/start',data=b'{}',
                                               headers={'Host':allowed,'Origin':f'http://{allowed}','Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(request,timeout=2)
                self.assertEqual(error.exception.code,403)
                self.assertFalse(list(Path(folder).glob('run-*/jobs/*/job.json')))
            finally:
                p.terminate();p.wait(timeout=10)
if __name__=='__main__':unittest.main()
