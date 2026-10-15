"""Single-component IM and family selection. Pair IDs are provenance, never responses."""
import math
from common import ROOT,resolve,sha

def select_components(motions,modes):
    from fragility_poc.ground_motion import load_acceleration,spectral_acceleration_g
    periods={axis:float(modes[axis]['modes'][0]['period_s']) for axis in ('X','Y')}
    tref=math.sqrt(periods['X']*periods['Y']);families={}
    for m in motions:
        if m.get('valid',1) and m['source_set']=='CMS_ZONE5':families.setdefault(float(m['conditioning_period_s']),[]).append(m)
    eligible={}
    for period,rows in families.items():
        unique={r.get('physical_pair_hash') or (r['sha256_x'],r['sha256_y']):r for r in rows}
        if len(unique) in (3,4):eligible[period]=sorted(unique.values(),key=lambda r:r['pair_id'])
    if not eligible:raise ValueError('No CMS family with 3 or 4 verified physical pairs')
    family=min(eligible,key=lambda p:(abs(p-tref),p));components=[]
    for motion in eligible[family]:
        for axis in ('X','Y'):
            path=resolve(motion[f'component_{axis.lower()}_path'])
            checksum=motion[f'sha256_{axis.lower()}']
            if sha(path)!=checksum:raise ValueError('GM component checksum mismatch')
            t,acc=load_acceleration(path,motion['units'])
            sa=float(spectral_acceleration_g(acc,float(motion['dt_s']),periods[axis],.05))
            if not math.isfinite(sa) or sa<=0:raise ValueError('Invalid component Sa(T1,5%)')
            components.append(dict(component_id=motion['pair_id']+'-'+axis,pair_id=motion['pair_id'],axis=axis,
                reference_period_s=periods[axis],unscaled_sa_g=sa,component_sha256=checksum,
                conditioning_period_s=family))
    return components,dict(policy='nearest single installed CMS family, no midpoint blend',family_s=family,
        family_selection_period_s=tref,axis_periods_s=periods,pairs=len(eligible[family]),components=len(components),
        outside_family_period_range=tref<min(eligible) or tref>max(eligible),
        IM_definition='Sa_component(T1_direction,5%)',independent_event_count_claim=False)
