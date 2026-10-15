"""Bounded best-feasible selection and complete Development8, floors6–9.

Started pilot studies stay immutable. This separate admission experiment measures
all original development pairs under the user's80% crossing-only policy.
Existing valid Full3D histories may end later than4%: retain their original
statuses/peaks and use observed threshold crossings, never reinterpret failures.
"""
import argparse, copy, datetime, hashlib, json, os, sys, time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'.build'))
import quick_development_screen as q
import development_next_stage_v1 as engine
ORIGINAL_OUT=engine.OUT
OUT=q.OUT/'consistent-selected-v1'
DECL=OUT/'preregistration.json'
THRESHOLDS=engine.THRESHOLDS
VARIANTS={6:'mesh_spatial_elastic',7:'reference_mesh',8:'mesh_spatial_elastic',9:'reference_mesh'}
read,save,sha=q.read,q.save,q.sha

def folder(j):return OUT/'cases'/j['id']
def terminal(j):
    p=folder(j)/'status.json'
    return p.exists() and read(p).get('state') in ('complete','failed','needs_review')
def result(j):
    p=folder(j)/'result.json'
    return read(p) if p.exists() else None

def prepare():
    if DECL.exists():raise RuntimeError('Started declaration is immutable')
    import cross_match_completion as cm
    audit=read(q.OUT/'development_audit.json');old_report=read(q.OUT/'modal-spo-report-v1/preregistration.json')
    sources={};jobs=[];decisions=[];hashes={}
    def freeze(p):
        p=Path(p).resolve();sources[str(p)]=sha(p)
    for p in (Path(__file__),BASE/'.build/development_next_stage_v1.py',BASE/'.build/autonomous_development_v1.py',
              q.OUT/'auto-v1/preregistration.json',q.OUT/'modal-spo-report-v1/preregistration.json',q.OUT/'development_audit.json',
              ORIGINAL_OUT/'preregistration.json',BASE/'.build/cp_endpoint_policy.py'):
        freeze(p)
    def reference(f,bid,part,name):
        for root in (Path(f['campaign']),Path(f['package'])/f'runs/{f["stories"]}storey/campaign'):
            p=root/'evidence'/bid/part/name
            if p.exists():freeze(p);return str(p)
        raise FileNotFoundError(f'{bid}/{part}/{name}')
    for f in audit['floors']:
        n=f['stories']
        if n==10:continue
        prior=read(q.OUT/f'auto-v1/cases/auto_dynamic_{n}_development_outlier/result.json');hashes[str(n)]=prior['candidate_hash']
        review=read(ORIGINAL_OUT/f'reviews/dynamic_{n}.json');freeze(ORIGINAL_OUT/f'reviews/dynamic_{n}.json')
        decisions.append(dict(stories=n,selected_variant=VARIANTS[n],selected_candidate_hash=hashes[str(n)],
            alternatives_review=str(ORIGINAL_OUT/f'reviews/dynamic_{n}.json'),challenger_supports_expansion=review['supports_expansion'],
            reason=('Elastic transverse pilot has materially better earlystatic than nonlinear challenger' if n==8 else
                    'Additional nonlinear components did not meaningfully improve fixed-IM outlier; retain simpler, statically supported prior recipe'),
            unresolved_development_outlier_error=prior['candidate_error'],perfect_outlier_agreement_claim=False,
            admission_policy='Complete fresh consistent Development8 and measure80%crossing; pilot5%improvement is not an additional final acceptance gate'))
        pkg=Path(f['package'])
        for name in (f'config/{n}storey.json','scripts/campaign_worker.py','scripts/run_fishbone_v5_dense_ida_case.py',
                     'src/fragility_poc/fishbone_v4_grouped_frame.py','src/fragility_poc/fishbone_cyclic.py','src/fragility_poc/fishbone_spo.py'):
            freeze(pkg/name)
        for bid in f['development_ids']:
            for axis in ('X','Y'):
                for mode in (1,2,3):
                    j=dict(id=f'selected{n}_static_{bid}_{axis}_mode{mode}',stories=n,kind='static',building_id=bid,
                        direction=axis,mode=mode,variant=VARIANTS[n],
                        reference=reference(f,bid,'static',f'{axis}_mode{mode}_full3d.npz'),
                        baseline=reference(f,bid,'static',f'{axis}_mode{mode}_fishbone.npz'))
                    old=next((x for x in old_report['new_jobs']+old_report['reused_jobs'] if x['stories']==n and x['building_id']==bid and x['direction']==axis and x['mode']==mode),None)
                    if old:
                        p=Path(old['result_path']) if 'result_path' in old else q.OUT/'modal-spo-report-v1/cases'/old['id']/'result.json'
                        if p.exists():j['reuse_result']=str(p);freeze(p);freeze(read(p)['history_path'])
                    jobs.append(j)
                for pattern in ('first_mode','uniform'):
                    row=next(x for x in f['cyclic'] if x['building_id']==bid and x['direction']==axis and x['pattern']==pattern)
                    jobs.append(dict(id=f'selected{n}_cyclic_{bid}_{axis}_{pattern}',stories=n,kind='cyclic',building_id=bid,
                        direction=axis,pattern=pattern,variant=VARIANTS[n],reference_status=row['reference_status'],
                        reference=reference(f,bid,'cyclic',f'{axis}_{pattern}_full3d.npz')))
        fulljobs=[]
        if n<8:
            for bid in f['development_ids']:
                p=Path(f['campaign'])/'evidence'/bid/'sparse.json';signed=read(p);freeze(p)
                for row in signed['cases']:
                    if row['role'] not in THRESHOLDS:continue
                    fulljobs.append(dict(building_id=bid,pair_id=row['pair_id'],role=row['role'],im_g=row['im_g'],
                        reference_payload=row['full3d'],source_metadata=str(p),source_kind='signed_endpoint_development',source_job=None))
        else:
            matrix=read(cm.OUT/f'matrix_{n}.json');freeze(cm.OUT/f'matrix_{n}.json')
            for old in matrix['jobs']:
                if old['engine']!='full3d':continue
                payload=cm.cached(old);source=old.get('assembled_source')
                source_path=source['path'] if source else old['cache'] if Path(old['cache']).exists() else None
                source_kind='unchanged_reference_point'
                # The baseline8 complete signed development already supplies these exact IMs.
                # Its15%termination can differ; original peak/status remain untouched and
                # boolean1/2/4%crossings remain directly observed on unchanged physics.
                if payload is None and n==8:
                    p=Path(f['package'])/f'runs/{n}storey/campaign/evidence'/old['building_id']/'sparse.json'
                    if p.exists():
                        signed=read(p);row=next((r for r in signed['cases'] if (r['pair_id'],r['role'],r['im_g'])==(old['pair_id'],old['role'],old['im_g'])),None)
                        if row:payload=row['full3d'];source_path=str(p);source_kind='signed_baseline_same_physics_later_termination'
                fulljobs.append(dict(building_id=old['building_id'],pair_id=old['pair_id'],role=old['role'],im_g=old['im_g'],
                    reference_payload=payload,source_metadata=source_path,source_kind=source_kind,source_job=old))
        if len(fulljobs)!=96 or len({(j['building_id'],j['pair_id'],j['role'],j['im_g']) for j in fulljobs})!=96:
            raise RuntimeError('Incomplete original Development8 matrix')
        for row in fulljobs:
            if row['building_id'] not in f['development_ids']:raise RuntimeError('Holdout forbidden')
            token=hashlib.sha256(json.dumps([n,row['building_id'],row['pair_id'],row['role'],row['im_g']]).encode()).hexdigest()[:24]
            ref=dict(id=f'selected{n}_reference_{token}',stories=n,kind='reference',**row)
            payload=row['reference_payload']
            if payload:
                if row['source_metadata']:freeze(row['source_metadata'])
                path=payload.get('history_path',payload.get('mdof_history_path'))
                if path and Path(path).exists():freeze(path)
                if payload.get('building_id') not in (None,row['building_id']) or payload.get('pair_id') not in (None,row['pair_id']):raise RuntimeError('Reference identity mismatch')
                if payload.get('target_im_g') is not None and abs(payload['target_im_g']-row['im_g'])>1e-10:raise RuntimeError('Reference IM mismatch')
            jobs.append(ref)
            j=dict(id=f'selected{n}_dynamic_{token}',stories=n,kind='dynamic',building_id=row['building_id'],pair_id=row['pair_id'],
                role=row['role'],im_g=row['im_g'],variant=VARIANTS[n],reference_job=ref['id'])
            for p in (q.OUT/'auto-v1/cases').glob(f'auto_dynamic_{n}_*/result.json'):
                pr=read(p);pj=pr['job']
                if (pj['building_id'],pj['pair_id'],pj['im_g'])==(j['building_id'],j['pair_id'],j['im_g']):
                    j['reuse_result']=str(p);freeze(p);freeze(pr['history_path']);break
            jobs.append(j)
    ids={str(f['stories']):f['development_ids'] for f in audit['floors'] if f['stories']<10}
    d=dict(schema='bounded-best-feasible-consistent-development-v1',declared_at=datetime.datetime.now().astimezone().isoformat(),
        human_policy='Best feasible tuning then operational80% crossing admission; NO dense qualification; build500only',
        source_hashes=sources,jobs=jobs,development_ids=ids,decisions=decisions,candidate_hashes=hashes,
        static_runtime_s=300,static_maximum_story_drift=.02,development_cyclic_protocol=[.0025,.005,.01,.015,.02,.03,.04],
        development_cyclic_runtime_s=900,dynamic_runtime_s_per_direction=1200,dynamic_cp_threshold=.04,
        original_signed_gate_retry=False,new_measurement_scope='Prior candidates have only2freshpilot cases; measure all8freshconsistent candidate comparisons under new explicit operational admission, not rerun old signedbaselinegates',
        reference_reuse_policy='Same building/pair/IM and physical model; original statuses/peaks and late termination retained. Only crossing/capped metrics, noclaim identical uncappedpeak orfreshreference solve. Numericalfailures are never promoted.',
        numerical_failure_relabel_allowed=False,physical_collapse_claim=False,holdout_used=False,
        scientific_qualification_passed=False,dense_qualification_enabled=False,population_analysis_enabled=False,
        operational_cross_match=.8,expected_per_height=dict(static=48,cyclic=32,dynamic=96,reference=96),
        build_only_destination=str(q.OUT/'prepare-only-2026-10-11'))
    save(DECL,d);schedule=[]
    for j in jobs:
        if j.get('reuse_result'):
            v=read(j['reuse_result']);h=hashes[str(j['stories'])]
            if v.get('candidate_hash') not in (None,h):raise RuntimeError('Reused candidate identity differs')
            if v.get('candidate_hash') is None:v['candidate_identity_provenance']='Same source model/fiber16 constructor as auto-v1; original unrecorded hash retained in source'
            v.update(candidate_hash=h,job=j,reused_from=j['reuse_result'],reused_result_sha256=sha(j['reuse_result']))
        elif j['kind']=='reference' and j['reference_payload'] is not None:v=dict(job=j,payload=j['reference_payload'],reference_reused=True,source_kind=j['source_kind'])
        else:
            schedule.append(dict(id=j['id'],kind='runtime_study',package='6-7 floor' if j['stories']<8 else '8-10 floor',
                entry='.build/consistent_selected_development_v1.py',arguments=['--action','case','--job',j['id']],
                status_path=str((folder(j)/'status.json').relative_to(BASE))));continue
        save(folder(j)/'result.json',v);save(folder(j)/'status.json',dict(state='complete',reused=True))
    save(BASE/'.build/consistent_selected_development_plan_v1_2026-10-11.json',schedule)
    save(OUT/'no_solver_validation.json',dict(solvers_started=0,unique_jobs=len({j['id'] for j in jobs})==len(jobs),
        total_jobs=len(jobs),reused=len(jobs)-len(schedule),scheduled=len(schedule),
        new_full3d_jobs=sum(j['kind']=='reference' and j['reference_payload'] is None for j in jobs),
        all_ids_development=True,old_failed_gates_untouched=True,no_population_analysis=True,alias_reference_mesh_to_mesh16=True))
    print(json.dumps(read(OUT/'no_solver_validation.json')))

def checked():
    d=read(DECL)
    for p,h in d['source_hashes'].items():
        if sha(p)!=h:raise RuntimeError('Frozen selected-development source/reference changed: '+p)
    return d

def inputs(j):
    import autonomous_development_v1 as auto
    cw,c,candidate,a,b=auto.inputs(j['stories'],j['building_id'],j['variant'])
    if candidate['freeze_hash']!=read(DECL)['candidate_hashes'][str(j['stories'])]:raise RuntimeError('Selected candidate changed')
    c['run_dir']=str(OUT/f'runs/{j["stories"]}storey');c['output_dir']=str(OUT/f'outputs/{j["stories"]}storey')
    return cw,c,candidate,a,b

def solve_reference(j,d):
    import cross_match_completion as cm
    cached=cm.cached(j['source_job']) if j.get('source_job') else j['reference_payload']
    if cached is not None:return dict(job=j,payload=cached,reference_reused=True)
    cw=q.context(j['stories']);c,m,root,b=cw.context(j['stories'],j['building_id'])
    if j['building_id'] not in m['development_ids']:raise RuntimeError('Reference outside development')
    with cw.connect(c['database_path']) as conn:pair=dict(conn.execute('SELECT * FROM ground_motion_catalog WHERE pair_id=?',(j['pair_id'],)).fetchone())
    from fragility_poc.fishbone_validation import ensure_full3d_case
    r=ensure_full3d_case(b,pair,j['im_g'],c)
    return dict(job=j,payload=cw.json_values(r['payload']),reference_reused=r['cache_reused'])

def install():
    engine.OUT=OUT;engine.DECL=DECL;engine.inputs=inputs;engine.solve_reference=solve_reference

def case(id):
    d=checked();j=next(j for j in d['jobs'] if j['id']==id)
    if result(j):return
    install();save(folder(j)/'status.json',dict(state='running',pid=os.getpid(),started_at=time.time()))
    action={'static':engine.solve_static,'cyclic':engine.solve_cyclic,'dynamic':engine.solve_dynamic,'reference':solve_reference}[j['kind']]
    # Shared cyclic helper recognises development_cyclic as the full protocol.
    solvej=dict(j,kind='development_'+j['kind'])
    try:
        v=action(solvej,d);v.update(job=j,holdout_used=False,population_analysis=False,scientific_qualification_passed=False)
        save(folder(j)/'result.json',v);save(folder(j)/'status.json',dict(state='complete',completed_at=time.time(),pid=os.getpid(),result_status=v.get('status')))
    except BaseException as exc:
        save(folder(j)/'status.json',dict(state='needs_review',error=repr(exc),pid=os.getpid()));raise

def report(n,d):
    import math
    jobs=[j for j in d['jobs'] if j['stories']==n];groups={k:[j for j in jobs if j['kind']==k] for k in ('static','cyclic','dynamic','reference')}
    coverage={k:dict(expected=len(v),terminal=sum(terminal(j) for j in v),results=sum(result(j) is not None for j in v)) for k,v in groups.items()}
    refs={j['id']:result(j) for j in groups['reference']};rows=[]
    for j in groups['dynamic']:
        g=result(j);r=refs[j['reference_job']];f=r['payload'] if r else None
        fp=(f or {}).get('max_midr',(f or {}).get('maximum_midr'));gp=(g or {}).get('candidate_midr');path=(f or {}).get('history_path',(f or {}).get('mdof_history_path'))
        valid=bool(g and g.get('usable') and f and not f.get('numerical_failure') and f.get('status') in ('success','cp_endpoint_reached','dynamic_instability') and isinstance(fp,(int,float)) and math.isfinite(fp))
        if valid:
            if not path or not Path(path).exists():valid=False
            else:
                peak,proof=engine.history_peak(path)
                if abs(peak-fp)>1e-8:raise RuntimeError('Reference raw peak mismatch')
                if f['status'] in ('cp_endpoint_reached','dynamic_instability') and proof is None:valid=False
        t=THRESHOLDS[j['role']]
        rows.append(dict(job=j,full3d_status=(f or {}).get('status'),grouped_status=(g or {}).get('status'),full3d_midr=fp,grouped_midr=gp,
            usable=valid,crossing_match=((fp>=t)==(gp>=t)) if valid else None,
            capped_relative_error=abs(min(gp,t)-min(fp,t))/max(min(fp,t),1e-12) if valid else None,
            full3d_history=path,grouped_history=(g or {}).get('history_path'),source_kind=(r or {}).get('source_kind',(r or {}).get('job',{}).get('source_kind')),
            terminal=terminal(j) and terminal(next(x for x in groups['reference'] if x['id']==j['reference_job']))))
    crossing={}
    for role in THRESHOLDS:
        rr=[r for r in rows if r['job']['role']==role];vv=[r for r in rr if r['usable']];matched=sum(r['crossing_match'] is True for r in vv)
        crossing[role]=dict(matched=matched,total=len(vv),expected=len(rr),terminal=sum(r['terminal'] for r in rr),
            invalid_terminal=sum(r['terminal'] and not r['usable'] for r in rr),match_rate=matched/len(vv) if vv else None,match_over_expected=matched/len(rr))
    complete=all(v['expected']==v['results'] for v in coverage.values());passed=complete and all(v['total'] and v['matched']/v['total']>=.8 for v in crossing.values())
    review=dict(stories=n,state='operational_pass' if passed else 'complete_below80' if complete else 'running',development_complete=complete,
        development_ids=d['development_ids'][str(n)],candidate_hash=d['candidate_hashes'][str(n)],cross_match=crossing,coverage=coverage,rows=rows,
        scientific_qualification_passed=False,holdout_used=False,dense_qualification_enabled=False,population_analysis_enabled=False,physical_collapse_claim=False)
    save(OUT/f'reviews/{n}storey.json',review)
    if passed:
        dest=Path(d['build_only_destination'])/f'accepted/{n}storey'
        if not (dest/'accepted_recipe.json').exists():
            import autonomous_development_v1 as auto
            cw,c,candidate,a,b=auto.inputs(n,d['development_ids'][str(n)][0],VARIANTS[n])
            variant='mesh16' if VARIANTS[n]=='reference_mesh' else VARIANTS[n]
            recipe=dict(stories=n,variant=variant,development_variant=VARIANTS[n],candidate_hash=candidate['freeze_hash'],model_config=c['model'],
                building_config=c['building'],ida_config=c['ida'],candidate_definition=candidate,source_hashes={str(DECL):sha(DECL),str(Path(__file__).resolve()):sha(__file__)},
                build_only=True,operational_threshold=.8,scientific_qualification_passed=False)
            save(dest/'accepted_recipe.json',recipe);review['recipe_sha256']=sha(dest/'accepted_recipe.json');save(dest/'development_review.json',review)
    return review

def observe():
    d=checked()
    while True:
        summaries=[]
        for n in range(6,10):
            # Separate process perheight prevents imported package/constructor aliases mixing.
            import subprocess
            pkg=BASE/('6-7 floor' if n<8 else '8-10 floor')
            with (OUT/f'report_{n}.stdout.log').open('a',encoding='utf-8') as o,(OUT/f'report_{n}.stderr.log').open('a',encoding='utf-8') as e:
                code=subprocess.run([str(pkg/'runtime/python.exe'),'-X','utf8',str(Path(__file__).resolve()),'--action','report','--stories',str(n)],cwd=BASE,stdout=o,stderr=e).returncode
            if code:raise RuntimeError(f'IO report{n} failed; inspect stderr')
            r=read(OUT/f'reviews/{n}storey.json');summaries.append({k:r[k] for k in ('stories','state','coverage','cross_match')})
        save(OUT/'pipeline_status.json',dict(state='complete' if all(terminal(j) for j in d['jobs']) else 'running',heights=summaries,
            pid=os.getpid(),updated_at=time.time(),numerical_solver=False,population_analysis=False))
        if all(terminal(j) for j in d['jobs']):return
        time.sleep(60)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','case','report','observe'),required=True);p.add_argument('--job');p.add_argument('--stories',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='case':case(a.job)
    elif a.action=='report':print(json.dumps(report(a.stories,checked())['coverage']))
    else:observe()
