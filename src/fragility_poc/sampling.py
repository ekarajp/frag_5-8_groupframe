"""Feasibility-certified geometry/strength transportation quotas for R9."""
from collections import Counter,defaultdict
import random
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp

def balanced_transport_sample(rows,size,seed):
    from .catalog import _capacity_constrained_equal_quotas
    if size>len(rows):raise ValueError('Insufficient feasible unique buildings')
    rng=random.Random(seed);ties={r['building_id']:rng.random() for r in rows}
    bays=sorted({int(r['number_of_bays']) for r in rows})
    bay_counts=Counter(int(r['number_of_bays']) for r in rows)
    bay_quota=_capacity_constrained_equal_quotas(bays,dict(bay_counts),size)
    geometries=sorted({(int(r['number_of_bays']),float(r['bay_width_m'])) for r in rows})
    geo_counts=Counter((int(r['number_of_bays']),float(r['bay_width_m'])) for r in rows)
    geo_quota={}
    for bay_index,bay in enumerate(bays):
        cells=[g for g in geometries if g[0]==bay]
        shift=bay_index%len(cells);cells=cells[shift:]+cells[:shift]
        geo_quota.update(_capacity_constrained_equal_quotas(cells,{g:geo_counts[g] for g in cells},bay_quota[bay]))
    tiers=sorted({(int(r['beam_tier']),int(r['column_tier'])) for r in rows})
    tier_counts=Counter((int(r['beam_tier']),int(r['column_tier'])) for r in rows)
    tier_quota=_capacity_constrained_equal_quotas(tiers,dict(tier_counts),size)
    def cell(r):return ((int(r['number_of_bays']),float(r['bay_width_m'])),(int(r['beam_tier']),int(r['column_tier'])))
    pools=defaultdict(list)
    for row in rows:pools[cell(row)].append(row)
    cells=sorted(pools);k=len(cells)
    # Integer x allocates actual models; continuous d measures deviation from
    # the balanced cross-table target. Every hard marginal is an equality.
    matrix=np.zeros((len(geometries)+len(tiers),2*k));supply=[]
    for i,g in enumerate(geometries):
        for j,(gg,t) in enumerate(cells):matrix[i,j]=int(gg==g)
        supply.append(geo_quota[g])
    for i,t in enumerate(tiers,len(geometries)):
        for j,(g,tt) in enumerate(cells):matrix[i,j]=int(tt==t)
        supply.append(tier_quota[t])
    target=np.array([geo_quota[g]*tier_quota[t]/size for g,t in cells])
    deviation=np.block([[np.eye(k),-np.eye(k)],[-np.eye(k),-np.eye(k)]])
    result=milp(c=np.r_[np.array([rng.random()*1e-7 for _ in cells]),np.ones(k)],
        integrality=np.r_[np.ones(k),np.zeros(k)],
        bounds=Bounds(np.zeros(2*k),np.r_[[len(pools[x]) for x in cells],np.full(k,np.inf)]),
        constraints=[LinearConstraint(matrix,supply,supply),LinearConstraint(deviation,np.full(2*k,-np.inf),np.r_[target,-target])],
        options={'time_limit':60,'mip_rel_gap':0.0})
    if result.x is None:raise RuntimeError('No feasible joint geometry/strength allocation; inspect catalogue availability. '+result.message)
    quotas={cell:int(round(value)) for cell,value in zip(cells,result.x[:k])}
    if sum(quotas.values())!=size or any(quotas[x]<0 or quotas[x]>len(pools[x]) for x in cells):raise RuntimeError('Invalid integer sampling solution')
    for g in geometries:
        if sum(q for (gg,t),q in quotas.items() if gg==g)!=geo_quota[g]:raise RuntimeError('Geometry quota mismatch')
    for t in tiers:
        if sum(q for (g,tt),q in quotas.items() if tt==t)!=tier_quota[t]:raise RuntimeError('Strength quota mismatch')
    fields=('fc_ksc','sdl_kg_m2','ll_kg_m2','scwb_class','base_case_id')
    counts=Counter();used=Counter();selected=[]
    availability={key:Counter(r[key] for r in rows) for key in fields}
    targets={key:_capacity_constrained_equal_quotas(sorted(values,key=str),dict(values),size) for key,values in availability.items()}
    remaining=list(rows)
    while len(selected)<size:
        eligible=[r for r in remaining if used[cell(r)]<quotas[cell(r)]]
        def score(r):
            coverage=used[cell(r)]/max(quotas[cell(r)],1)
            imbalance=sum(counts[(f,r[f])]/max(targets[f][r[f]],1) for f in fields)
            return (coverage,imbalance,ties[r['building_id']])
        chosen=min(eligible,key=score);remaining.remove(chosen);selected.append(chosen);used[cell(chosen)]+=1
        for f in fields:counts[(f,chosen[f])]+=1
    return selected
