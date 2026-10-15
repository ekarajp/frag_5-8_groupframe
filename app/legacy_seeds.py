"""Reuse complete original directional observations without changing any native flags."""
import math,shutil
from common import ROOT,read,write,sha,digest,resolve
def import_seeds(job,run,out,recipe):
    marker=out/'legacy_import.json'
    if marker.exists():return
    import numpy as np
    from fragility_poc.ground_motion import load_acceleration,spectral_acceleration_g
    from worker import validate_history
    index=read(ROOT/'assets/legacy_seed_index.json');sources=index.get(str(job['model']['stories'])+'--'+job['model']['building_id']+'--'+job['component_id'],[])
    accepted=[];excluded=[];seen=set();checked_gm={}
    for source in sources:
        metadata=resolve(source['metadata']);history=resolve(source['history'])
        assert sha(metadata)==source['metadata_sha256'] and sha(history)==source['history_sha256']
        r=read(metadata);j=r['job'];axis=job['axis']
        assert j['building_id']==job['model']['building_id'] and j['direction']==axis and j['pair_id']==job['pair_id']
        assert r['candidate_hash']==recipe['candidate_hash']==job['model']['candidate_hash']
        if r['numerical_failure']:
            raise ValueError('Original signed component failure requires engineering review; automatic unchanged retry refused')
        # Cross-check the original paired spectral ordinate against the frozen
        # waveform snapshot before mapping its actual amplitude into component IM.
        original_period=float(r['signature']['reference_period_s'])
        if original_period not in checked_gm:
            motion=next(m for m in read(ROOT/'assets/motions.json') if m['pair_id']==job['pair_id'])
            ordinates=[]
            for component in ('x','y'):
                wave=resolve(motion['component_'+component+'_path'])
                assert sha(wave)==motion['sha256_'+component]
                _,acc=load_acceleration(wave,motion['units'])
                ordinates.append(float(spectral_acceleration_g(acc,motion['dt_s'],original_period,.05)))
            checked_gm[original_period]=math.sqrt(ordinates[0]*ordinates[1])
        if not math.isclose(checked_gm[original_period],r['signature']['unscaled_sa_g'],rel_tol=1e-7,abs_tol=1e-10):
            raise ValueError('Original observation belongs to another GM snapshot; import refused')
        period=r['damping']['reported_directional_modes'][0]['period_s']
        if not math.isclose(period,job['reference_period_s'],rel_tol=1e-8,abs_tol=1e-10):
            excluded.append(dict(source=source,reason='Directional modal period differs; not reused'));continue
        with np.load(history) as z:peak=validate_history(dict(z),job['model']['stories'])
        valid=not r['numerical_failure'] and (r['completed_fraction']>=1-1e-8 or peak>=.04)
        if not valid:
            excluded.append(dict(source=source,reason='Original native failure or incomplete non-CP observation'));continue
        sf=r['signature']['scale_factor'];im=sf*job['unscaled_sa_g']
        if (sf,source['history_sha256']) in seen:continue
        seen.add((sf,source['history_sha256']))
        accepted.append((im,r,source,peak))
    # A failed partial import is not automatically replayed over existing output.
    if any(out.glob('point-*')) and sources:raise ValueError('Unregistered legacy import output exists; audit before resuming')
    for i,(im,r,source,peak) in enumerate(sorted(accepted,key=lambda x:x[0])):
        pointdir=out/f'point-{i:03d}';pointdir.mkdir();axis=job['axis']
        shutil.copy2(resolve(source['history']),pointdir/(axis+'.npz'));shutil.copy2(resolve(source['metadata']),pointdir/'legacy_original_result.json')
        native={k:v for k,v in r.items() if k not in ('job','history_path')};write(pointdir/(axis+'.json'),native)
        hashes={name:sha(pointdir/name) for name in (axis+'.npz',axis+'.json','legacy_original_result.json')}
        status='cp_endpoint_reached' if peak>=.04 else 'success'
        p=dict(schema='portable-component-IDA-point-v2',component_id=job['component_id'],pair_id=job['pair_id'],axis=axis,
            target_im_g=im,scale_factor=r['signature']['scale_factor'],unscaled_sa_g=job['unscaled_sa_g'],reference_period_s=job['reference_period_s'],
            max_midr=peak,status=status,numerical_failure=False,completed_fraction=r['completed_fraction'],
            native_directional_status=r['status'],physical_collapse_claim=False,committed_cp_proof=peak>=.04,
            file_hashes=hashes,run_hash=run['run_hash'],model_sha256=job['model']['sha256'],candidate_hash=recipe['candidate_hash'],
            gm_hashes=[job['component_sha256']],point_index=i,IM_definition='Sa_component(T1_direction,5%)',
            reused_original_directional_observation=True,original_pair_signature=r['signature'],
            source_metadata_sha256=source['metadata_sha256'],source_history_sha256=source['history_sha256'],
            member_envelopes='not_recorded_in_original_observation',native_flags_preserved=True)
        p['case_signature']=digest({k:p[k] for k in ('run_hash','component_id','target_im_g','scale_factor','reference_period_s','model_sha256','candidate_hash','gm_hashes','source_history_sha256')})
        write(pointdir/'component.json',p)
    write(marker,dict(accepted=len(accepted),excluded=excluded,source_observations=len(sources),native_analyses=0))
