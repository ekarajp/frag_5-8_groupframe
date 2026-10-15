"""Report dependencies and optionally import native engine, without analyze()."""
import argparse,importlib,importlib.metadata,json,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from common import environment,identity
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--solver-import',action='store_true');args=parser.parse_args()
    assert sys.version_info[:2]==(3,10),'Use Python 3.10 x64, or Docker linux/amd64.'
    assert platform.system() in ('Windows','Linux'),'On macOS use Docker linux/amd64 with the frozen solver version.'
    for name in ('numpy','scipy','matplotlib','psutil'):importlib.import_module(name)
    report=environment();report['process_identity']=identity(__import__('os').getpid());assert report['process_identity']
    expected='3.5.1.3'
    assert importlib.metadata.version('openseespy')==expected
    native='openseespywin' if sys.platform=='win32' else 'openseespylinux'
    assert importlib.metadata.version(native)==expected
    if args.solver_import:
        import openseespy.opensees as ops
        report['native_engine_version']=ops.version()
    report['analysis_calls']=0;report['native_import_requested']=args.solver_import
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
