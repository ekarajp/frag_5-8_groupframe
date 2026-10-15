"""Source routing and Unicode/C-drive runtime relocation, zero native calls."""
import sys,subprocess,json,tempfile,shutil,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from common import ROOT,write
def probe(script,story):
    p=subprocess.run([sys.executable,'-B','-X','utf8',str(script),'--stories',str(story)],capture_output=True,timeout=120)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    result=json.loads(p.stdout);assert result['native_analyses']==0 and not result['solver_imported'];return result
def main():
    results=[probe(ROOT/'tests/source_route_probe.py',n) for n in (5,6,7,8)]
    base=Path(tempfile.gettempdir()).resolve();base.mkdir(exist_ok=True)
    target=Path(tempfile.mkdtemp(prefix='component58_IO_ไทย_',dir=base)).resolve()
    assert target.is_relative_to(base) and target.name.startswith('component58_IO_')
    try:
        for name in ('src','src_variants'):
            shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('__pycache__'))
        (target/'app').mkdir();shutil.copy2(ROOT/'app/common.py',target/'app/common.py')
        (target/'tests').mkdir();shutil.copy2(ROOT/'tests/source_route_probe.py',target/'tests/source_route_probe.py')
        relocated=[probe(target/'tests/source_route_probe.py',n) for n in (5,6)]
        assert all(Path(r['root'])==target and Path(r['runtime_storage'])==target/'runtime_storage' for r in relocated)
    finally:
        assert target.resolve().is_relative_to(base) and target.name.startswith('component58_IO_')
        shutil.rmtree(target)
    write(ROOT/'validation/source_routing_and_relocation.json',dict(passed=True,original_location=results,relocated_C_drive_Unicode=relocated,native_analyses=0,scope='Source imports and runtime/cache location only; no solver/analyses'))
    print('SOURCE_ROUTING_AND_UNICODE_RELOCATION_PASSED; 0 native analyses',flush=True)
if __name__=='__main__':main()
