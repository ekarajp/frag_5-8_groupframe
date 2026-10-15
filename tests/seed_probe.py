"""Legacy namespaces, verified component IM and no repeated analyses."""
import sys,time,math,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from common import ROOT,read,write,sha,configure_scientific_source
configure_scientific_source(5)
import legacy_seeds
from fragility_poc.ground_motion import load_acceleration,spectral_acceleration_g
def main():
    catalog=read(ROOT/'assets/catalog.json');index=read(ROOT/'assets/legacy_seed_index.json')
    for key,sources in index.items():
        story,bid,cid=key.split('--',2)
        model=next((m for m in catalog if m['stories']==5 and m['building_id']==bid),None)
        if model:break
    assert model and story=='5'
    r=read(ROOT/sources[0]['metadata']);axis=r['direction'];period=r['damping']['reported_directional_modes'][0]['period_s']
    motion=next(m for m in read(ROOT/'assets/motions.json') if m['pair_id']==r['job']['pair_id'])
    wave=ROOT/motion['component_'+axis.lower()+'_path'];_,a=load_acceleration(wave,motion['units'])
    sa=float(spectral_acceleration_g(a,motion['dt_s'],period,.05))
    job=dict(model=model,component_id=cid,axis=axis,pair_id=motion['pair_id'],reference_period_s=period,unscaled_sa_g=sa,component_sha256=sha(wave))
    recipe=read(ROOT/model['recipe'])
    with tempfile.TemporaryDirectory() as d:
        out=Path(d);legacy_seeds.import_seeds(job,dict(run_hash='QA-no-native'),out,recipe)
        marker=read(out/'legacy_import.json');assert marker['accepted']==3
        for f in out.glob('point-*/component.json'):
            p=read(f);original=read(f.parent/'legacy_original_result.json')
            assert sha(f.parent/(axis+'.npz'))==original['history_sha256']==p['source_history_sha256']
            assert p['native_directional_status']==original['status']
            assert math.isclose(p['target_im_g'],original['signature']['scale_factor']*sa,rel_tol=1e-14)
        legacy_seeds.import_seeds(job,dict(run_hash='QA-no-native'),out,recipe)
        assert len(list(out.glob('point-*/component.json')))==3
    with tempfile.TemporaryDirectory() as d:
        other=dict(job,model=dict(model,stories=6))
        legacy_seeds.import_seeds(other,dict(run_hash='QA-no-native'),Path(d),recipe)
        assert read(Path(d)/'legacy_import.json')['accepted']==0
    def fail_read(path):
        x=read(path)
        if Path(path).name=='original_result.json':x['numerical_failure']=True
        return x
    with tempfile.TemporaryDirectory() as d,patch.object(legacy_seeds,'read',fail_read):
        try:legacy_seeds.import_seeds(job,{},Path(d),recipe)
        except ValueError as e:assert 'automatic unchanged retry refused' in str(e)
        else:raise AssertionError('An unchanged signed failure was ignored')
        assert not list(Path(d).glob('point-*'))
    assert not any(n.startswith('opensees') for n in sys.modules)
    result=dict(passed=True,accepted_example=3,raw_hashes_and_flags_preserved=True,component_IM_verified=True,idempotent=True,cross_height_import_rejected=True,signed_failure_retry_rejected=True,native_analyses=0)
    write(ROOT/'validation/legacy_reuse_validation.json',result)
    print('SEED_PROBE_PASSED 3 reused observations, no native analyses',flush=True)
if __name__=='__main__':main()
