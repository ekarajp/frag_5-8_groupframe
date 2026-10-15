"""Audit the delivered experiment and all retained file identities, no solver."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,read,sha,write,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--deep',action='store_true');args=p.parse_args()
    folder=Path(args.run).resolve() if args.run else ROOT/'state'/read(ROOT/'state/current.json')['run']
    r=read(folder/'run.json')
    assert digest({k:v for k,v in r.items() if k!='run_hash'})==r['run_hash'],'Frozen run hash mismatch'
    points,files=0,0
    errors=[]
    for pointpath in sorted(folder.glob('jobs/*/point-*/component.json')):
        point=read(pointpath);points+=1
        if point['run_hash']!=r['run_hash']:errors.append(str(pointpath)+': run identity')
        for name,checksum in point['file_hashes'].items():
            f=pointpath.parent/name;files+=1
            if not f.exists() or sha(f)!=checksum:errors.append(str(f)+': checksum')
        if args.deep:
            import numpy as np
            from worker import validate_history
            for axis in (point['axis'],):
                try:
                    with np.load(pointpath.parent/(axis+'.npz')) as z:
                        peak=validate_history({k:z[k] for k in z.files},z['story_drift_ratio'].shape[1])
                        import math
                        assert math.isclose(peak,point['max_midr'],rel_tol=1e-10,abs_tol=1e-12)
                        assert math.isclose(point['scale_factor']*point['unscaled_sa_g'],point['target_im_g'],rel_tol=1e-10)
                except Exception as e:errors.append(str(pointpath)+': '+str(e))
    inventory=folder/'research_inventory.json'
    if inventory.exists():
        for row in read(inventory)['files']:
            f=folder/row['path'];files+=1
            if not f.exists() or f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:
                errors.append(str(f)+': inventory')
    result={'run':str(folder),'points':points,'file_checks':files,'errors':errors,'deep':args.deep,
            'solver_imported':any(s.startswith('opensees') for s in sys.modules),'analysis_calls':0}
    write(ROOT/'validation/results_audit.json',result)
    print('RESULTS AUDIT',len(errors),'errors;',points,'points;',files,'file checks')
    if errors:sys.exit(1)

if __name__=='__main__':main()
