"""Import/routing/I/O probe only. Never calls OpenSees or an analysis."""
import sys,json,argparse,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from common import ROOT,configure_scientific_source,sha
p=argparse.ArgumentParser();p.add_argument('--stories',type=int,required=True);a=p.parse_args()
source=configure_scientific_source(a.stories)
import fragility_poc
from fragility_poc import structural,fishbone_v5_batched_dynamic,runtime_storage
assert Path(structural.__file__).resolve().is_relative_to(source.resolve())
storage=Path(os.environ['FRAGILITY_POC_RUNTIME_ROOT']).resolve()
assert storage==ROOT/'runtime_storage'
assert not any(n.startswith('opensees') for n in sys.modules)
print(json.dumps(dict(stories=a.stories,source=str(source),structural_sha256=sha(Path(structural.__file__)),runtime_storage=str(storage),root=str(ROOT),solver_imported=False,native_analyses=0)),flush=True)
