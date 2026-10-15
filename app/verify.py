"""Read-only, solver-free portability verification."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,read,write,sha,verify_assets,resolve

def main():
    import grouped_model_loader as loader
    import numpy as np
    checked=verify_assets()
    catalog=read(ROOT/'assets/catalog.json')
    counts={}
    for n in (5,6,7,8):
        models=[r for r in catalog if r['stories']==n]
        manifest=read(ROOT/f'assets/models/{n}storey/manifest.json')
        assert len(models)==500
        assert [r['building_id'] for r in models]==[r['building_id'] for r in manifest['models']]
        assert len({r['building_id'] for r in models})==500
        for row in models:
            m=loader.read_model(resolve(row['path']),row['sha256'])
            assert m['building_id']==row['building_id'] and m['source_model_hash']==row['model_hash']
            assert sha(resolve(row['recipe']))==m['recipe_sha256']
            class Recorder:
                def __init__(self):self.calls=[]
                def __getattr__(self,name):
                    if name not in loader.CONSTRUCTION_COMMANDS:raise ValueError(name)
                    return lambda *args:self.calls.append([name,list(args)])
            for axis in ('X','Y'):
                recorder=Recorder()
                loader.build_model(resolve(row['path']),axis,recorder,row['sha256'])
                assert recorder.calls==m['directions'][axis]['commands']
        counts[str(n)]=len(models)
    motions=read(ROOT/'assets/motions.json')
    assert len(motions)==23
    for row in motions:
        arrays=[np.loadtxt(resolve(row[f'component_{axis}_path'])) for axis in ('x','y')]
        assert all(np.isfinite(a).all() for a in arrays)
        assert np.allclose(arrays[0][:,0],arrays[1][:,0],rtol=0,atol=1e-10)
        assert np.all(np.diff(arrays[0][:,0])>0)
        assert row['source_set']=='CMS_ZONE5' and row['units']=='g'
    result={**checked,'models':counts,'X_Y_construction_replays':4000,'CMS_pairs':23,
            'analysis_calls':0,'solver_imported':any(n.startswith('opensees') for n in sys.modules)}
    assert not result['solver_imported']
    write(ROOT/'validation/package_verification.json',result)
    print('PACKAGE VERIFIED: 2000 models, 4000 construction replays, 23 CMS pairs, 0 analyses.')

if __name__=='__main__':main()
