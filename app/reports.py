"""Per-component curves/capacities; incomplete suites never receive a final fit."""
import csv,math,html
from pathlib import Path
from collections import Counter
import numpy as np
from common import ROOT,read,write,sha,THRESHOLDS

def fit_complete(capacities,parent_pairs=None,seed=68001500):
    values=np.asarray(capacities,float)
    if len(values)<3 or not np.isfinite(values).all() or np.any(values<=0):raise ValueError('At least 3 finite positive component capacities required')
    parents=list(parent_pairs) if parent_pairs is not None else list(range(len(values)))
    if len(parents)!=len(values):raise ValueError('Parent-pair count differs')
    groups=list(dict.fromkeys(parents))
    if len(groups)<3:raise ValueError('At least 3 parent pairs required for this fit')
    logs=np.log(values);rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(groups),size=(1000,len(groups)))
    samples=[np.concatenate([logs[[i for i,p in enumerate(parents) if p==groups[g]]] for g in row]) for row in draws]
    return dict(theta_g=float(np.exp(logs.mean())),beta_log_sample_ddof1=float(logs.std(ddof=1)),
        theta_95ci_g=np.exp(np.quantile([s.mean() for s in samples],[.025,.975])).tolist(),
        beta_95ci=np.quantile([s.std(ddof=1) for s in samples],[.025,.975]).tolist(),
        component_count=len(values),parent_pair_count=len(groups),bootstrap_replicates=1000,seed=seed,
        method='Log-capacity sample mean/sample standard deviation; bootstrap resamples whole parent pairs',
        uncertainty_scope='Between-record uncertainty with pair clustering; 6/8 components are not 6/8 independent earthquakes')
def probability(x,fit):
    from scipy.special import ndtr
    beta=fit['beta_log_sample_ddof1']
    return ndtr(np.log(x/fit['theta_g'])/beta) if beta>1e-12 else (x>=fit['theta_g']).astype(float)

def model_report(run_dir,model,jobs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=run_dir/'reports'/f"{model['stories']}-{model['building_id']}";dest.mkdir(parents=True,exist_ok=True)
    curve_jobs=[j for j in jobs if j.get('kind')=='curve'];curves=[]
    for job in curve_jobs:
        p=run_dir/'jobs'/job['id']/'curve.json'
        if not p.exists():continue
        c=read(p)
        if 'component_id' not in c or c.get('axis') not in ('X','Y'):raise ValueError('A paired curve cannot be silently imported as a component curve')
        for point in c['points']:
            folder=p.parent/f"point-{point['point_index']:03d}"
            for name,checksum in point['file_hashes'].items():
                if sha(folder/name)!=checksum:raise ValueError('Retained component history checksum mismatch')
        curves.append(c)
    expected=len(curve_jobs);parents={c['pair_id'] for c in curves}
    complete=expected in (6,8) and len(curves)==expected and len(parents)==expected//2 and all(
        c['state']=='complete' and all(cap['certified'] for cap in c['capacities'].values()) for c in curves)
    result=dict(model=model,expected_components=expected,complete_components=sum(c['state']=='complete' for c in curves),
        expected_parent_pairs=expected//2,state='complete' if complete else 'needs_review',
        IM_definition='Sa_component(T1_direction,5%)',thresholds_MIDR=THRESHOLDS,
        capacities={},fragility={},directional_fragility={},physical_collapse_claim=False,
        reused_observation_count=sum(bool(p.get('reused_original_directional_observation')) for c in curves for p in c['points']),
        selection='Nearest single CMS family; 4 pairs -> 8 component curves, 3 pairs -> 6 component curves',
        job_dispositions=[{k:j.get(k) for k in ('id','status','reason')} for j in jobs],
        limitations=['Original paired Development comparisons remain historical; no claim of new uniaxial Full3D qualification',
            'Numerical failure/guard is not a limit-state crossing; incomplete suite has no final fragility fit',
            'Parent pairs retained for clustered uncertainty; component count is not independent event count',
            'Floor histories at every committed state; member force/deformation envelopes; no full member or individual fiber histories'])
    with (dest/'ida_points.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['component_id','parent_pair_id','axis','T1_direction_s','IM_component_g','SF_component','MIDR_component','status','completed_fraction','reused_original_observation','source_history_sha256'])
        for c in curves:
            for p in c['points']:w.writerow([c['component_id'],c['pair_id'],c['axis'],c['reference_period_s'],p['target_im_g'],p['scale_factor'],p['max_midr'],p['status'],p['completed_fraction'],bool(p.get('reused_original_directional_observation')),p.get('source_history_sha256','')])
    with (dest/'component_capacities.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['component_id','parent_pair_id','axis','limit_state','capacity_component_g','lower_g','upper_g','certified'])
        for c in curves:
            for ls,cap in c.get('capacities',{}).items():w.writerow([c['component_id'],c['pair_id'],c['axis'],ls,cap.get('capacity_im_g'),cap.get('lower_im_g'),cap.get('upper_im_g'),cap.get('certified')])
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for c in curves:
        pts=sorted(c['points'],key=lambda p:p['target_im_g']);axes[0].plot([p['max_midr'] for p in pts],[p['target_im_g'] for p in pts],'.-',label=c['component_id'])
    for ls,threshold in THRESHOLDS.items():
        rows=[dict(component_id=c['component_id'],pair_id=c['pair_id'],axis=c['axis'],**c.get('capacities',{}).get(ls,{})) for c in curves]
        result['capacities'][ls]=rows;axes[0].axvline(threshold,color='grey',ls='--',lw=.6)
        if complete:
            fit=fit_complete([r['capacity_im_g'] for r in rows],[r['pair_id'] for r in rows]);result['fragility'][ls]=fit
            x=np.geomspace(min(r['capacity_im_g'] for r in rows)/3,max(r['capacity_im_g'] for r in rows)*3,200)
            axes[1].semilogx(x,probability(x,fit),label=ls)
            for axis in ('X','Y'):
                rr=[r for r in rows if r['axis']==axis]
                result['directional_fragility'].setdefault(axis,{})[ls]=fit_complete([r['capacity_im_g'] for r in rr],[r['pair_id'] for r in rr])
    axes[0].set(xlabel='Component MIDR',ylabel='Component Sa(T1 direction,5%) [g]',title=f'{expected} separate component IDA curves')
    if curves:axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Component Sa(T1 direction,5%) [g]',ylabel='P(exceedance)',ylim=(0,1),title='Component fragility / pair-cluster uncertainty' if complete else 'Incomplete: final fit withheld')
    if complete:axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"{model['stories']} storey / {model['building_id']}");fig.savefig(dest/'ida_fragility.png',dpi=150);plt.close(fig)
    write(dest/'results.json',result)
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Component IDA result</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto}img{width:100%}</style>'
        f'<h1>{model["stories"]} storey — {html.escape(model["building_id"])}</h1><p>{result["state"]} · {expected} component curves from {expected//2} pairs</p>'
        '<p>Each component has its own Sa(T1 direction), scale factor and IO/LS/CP capacity. No X/Y maximum or geometric-mean pair IM is used. Two components are not two independent events.</p>'
        '<img src="ida_fragility.png"><p><a href="results.json">JSON</a> · <a href="ida_points.csv">All evaluated points</a> · <a href="component_capacities.csv">Component capacities</a></p>',encoding='utf-8')
    write(dest/'manifest.json',dict(files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='manifest.json']))
    return result

def campaign_export(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=read(run_dir/'run.json');results=[read(p) for p in sorted((run_dir/'reports').glob('*/results.json'))]
    targets=plan.get('height_target_counts') or dict(Counter(str(r['stories']) for r in read(ROOT/'assets/catalog.json')))
    expected=plan['target_model_count'];good=[r for r in results if r['state']=='complete']
    summary=dict(schema='component-fragility-summary-v2',expected_models=expected,reported_models=len(results),certified_models=len(good),unresolved_models=expected-len(good),heights={},IM_definition=plan['IM_definition'])
    with (run_dir/'fragility_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['stories','building_id','state','limit_state','expected_components','complete_components','parent_pairs','theta_component_g','beta_log'])
        for r in results:
            for ls in THRESHOLDS:
                fit=r['fragility'].get(ls,{});w.writerow([r['model']['stories'],r['model']['building_id'],r['state'],ls,r['expected_components'],r['complete_components'],r['expected_parent_pairs'],fit.get('theta_g'),fit.get('beta_log_sample_ddof1')])
    for n,count in sorted(targets.items(),key=lambda x:int(x[0])):
        rs=[r for r in good if r['model']['stories']==int(n)];complete=len(rs)==count
        h=dict(expected=count,certified_models=len(rs),state='complete' if complete else 'incomplete_final_mixture_withheld',interpretation='Equal-building-weight mixture of component fragilities, not pooled independent events',probabilities={})
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained');x=np.geomspace(.001,20,250)
        if complete:
            for ls in THRESHOLDS:
                y=np.mean([probability(x,r['fragility'][ls]) for r in rs],axis=0);h['probabilities'][ls]=dict(im_component_g=x.tolist(),mean_probability=y.tolist());ax.semilogx(x,y,label=ls)
            ax.legend()
        else:ax.text(.5,.5,f'{len(rs)} / {count} models complete\nFinal mixture withheld',ha='center',va='center',transform=ax.transAxes)
        ax.set(xlabel='Component Sa at each directional T1 [g]',ylabel='Mean P(exceedance)',ylim=(0,1),title=f'{n} storey component fragility');ax.grid(alpha=.2);fig.savefig(run_dir/f'height-{n}-fragility.png',dpi=150);plt.close(fig);summary['heights'][n]=h
    write(run_dir/'campaign_summary.json',summary)
    links=''.join(f'<li><a href="reports/{r["model"]["stories"]}-{r["model"]["building_id"]}/index.html">{r["model"]["stories"]} / {html.escape(r["model"]["building_id"])}</a> — {r["state"]}</li>' for r in results)
    (run_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Component fragility campaign</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}img{max-width:100%}</style>'
        f'<h1>Component IDA / fragility</h1><p>{len(good)} / {expected} models complete. Numerical failure is not CP; incomplete suites are retained without final fits.</p>'
        '<p><a href="fragility_summary.csv">Summary CSV</a> · <a href="campaign_summary.json">Summary JSON</a> · <a href="research_inventory.json">SHA-256 inventory</a></p>'
        +''.join(f'<h2>{n} storey</h2><img src="height-{n}-fragility.png">' for n in targets)+'<ul>'+links+'</ul>',encoding='utf-8')
    write(run_dir/'research_inventory.json',dict(files=[dict(path=p.relative_to(run_dir).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(run_dir.rglob('*')) if p.is_file() and p.name not in ('research_inventory.json','queue.sqlite','queue.sqlite-wal','queue.sqlite-shm','events.jsonl','stop.flag')]))
