"""TEST ONLY: deterministic queue lifecycle, no OpenSees."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from common import read,write,sha,THRESHOLDS
p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--run');a=p.parse_args()
j,r=read(a.job),read(a.run);out=Path(a.job).parent
if j.get('simulate_failure'):
    print('QA simulated native failure before receipt',flush=True)
    sys.exit(13)
time.sleep(.12)
if j['kind']=='modal':
    write(out/'result.json',{'model':j['model'],'components':[dict(component_id=f'TEST-{i}-{axis}',pair_id=f'TEST-{i}',axis=axis,unscaled_sa_g=.2*i,reference_period_s=1.) for i in (1,2,3) for axis in ('X','Y')]})
    receipt={'state':'complete','result_sha256':sha(out/'result.json')}
else:
    if (out.parents[1]/'stop.flag').exists():
        receipt={'state':'paused'}
    else:
        capacities={k:{'certified':True,'status':'bracketed','capacity_im_g':v*20,
                          'lower_im_g':v*19.9,'upper_im_g':v*20.1} for k,v in THRESHOLDS.items()}
        write(out/'curve.json',{'state':'complete','points':[], 'capacities':capacities,
            'pair_id':j['pair_id'],'component_id':j['component_id'],'axis':j['axis'],'reference_period_s':1.})
        receipt={'state':'complete','curve_sha256':sha(out/'curve.json')}
write(out/'receipt.json',{**receipt,'job_id':j['id'],'run_hash':r['run_hash']})
