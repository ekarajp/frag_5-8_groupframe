import hashlib,importlib.util,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
import common
spec=importlib.util.spec_from_file_location('prepare_data',ROOT/'tools/prepare_data.py')
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
class PlatformTests(unittest.TestCase):
    def test_external_virtualenv_interpreter_is_recorded(self):
        with patch.object(sys,'executable',str(Path(tempfile.gettempdir())/'outside-venv/python')):
            self.assertTrue(common.environment()['executable'])
    def test_creation_identity_detects_terminated_process(self):
        import subprocess
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        try:
            ident=common.identity(p.pid)
            self.assertEqual(ident['pid'],p.pid);self.assertTrue(ident['created'])
            self.assertTrue(Path(ident['exe']).is_absolute())
        finally:p.terminate();p.wait(timeout=10)
        self.assertIsNone(common.identity(p.pid))
    def test_seed_install_is_idempotent_and_refuses_changed_existing_file(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);name='assets/legacy_seeds/case/history.npz';payload=b'fixture only'
            archive=root/'fixture.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr(name,payload)
            m=dict(bytes=archive.stat().st_size,sha256=data.sha(archive),files=[dict(path=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())])
            self.assertEqual(data.install(archive,m,root),1)
            self.assertEqual(data.install(archive,m,root),1)
            (root/name).write_bytes(b'changed')
            with self.assertRaises(AssertionError):data.install(archive,m,root)
    def test_seed_archive_traversal_rejected_without_write(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);name='assets/legacy_seeds/../../../escape'
            archive=root/'fixture.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr(name,b'x')
            m=dict(bytes=archive.stat().st_size,sha256=data.sha(archive),files=[dict(path=name,bytes=1,sha256=hashlib.sha256(b'x').hexdigest())])
            with self.assertRaises(AssertionError):data.install(archive,m,root)
            self.assertFalse((root/'escape').exists())
if __name__=='__main__':unittest.main()
