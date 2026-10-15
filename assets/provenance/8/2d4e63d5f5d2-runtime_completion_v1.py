"""Declared runtime completion and IO review overlay; immutable original cases."""
import argparse,copy,datetime,json,os,sys,time,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'.build'))
import consistent_selected_development_v1 as s
import development_next_stage_v1 as e
ROOT=BASE/'.build/development-tuning-2026-10-11/runtime-completion-v1';DECL=ROOT/'amendment.json'
read,save,sha=s.read,s.save,s.sha
ORIG={9:s.OUT,10:e.OUT}
def guard(r):
    dirs=r.get('original_directional_results',{})
    return bool(r.get('numerical_failure') and dirs and any('runtime_guard' in v.get('collapse_basis',[]) for v in dirs.values()))
def validate_result(r):
    p=Path(r['history_path']);assert sha(p)==r['history_sha256']
    peak,proof=e.history_peak(p);assert abs(peak-r['candidate_midr'])<1e-10
    assert r.get('physical_collapse_claim') is False
    return dict(history_sha256=sha(p),peak=peak,cp_proof=proof,usable=r['usable'])
def prepare():
    if DECL.exists():raise RuntimeError('Already frozen')
    d9=s.checked();d10=e.checked();entries=[];proofs=[]
    for n,d in ((9,d9),(10,d10)):
        probe=read(BASE/f'.build/development-tuning-2026-10-11/floor{n}-runtime-probe-v1/result.json')
        assert probe['usable'] and probe['status']=='success';proofs.append(dict(stories=n,**validate_result(probe)))
        for j in d['jobs']:
            if j['stories']!=n or j['kind'] not in ('dynamic','development_dynamic'):continue
            old=ORIG[n]/'cases'/j['id']/'result.json'
            if not old.exists() or not guard(read(old)):continue
            r=read(old);entry=dict(id='runtime_completion_'+j['id'],job=j,old_result=str(old),old_sha256=sha(old),candidate_hash=r['candidate_hash'],runtime_s=2400 if n==9 else 7200)
            if j['id']==probe['job']['id']:
                entry['reuse_probe']=str(BASE/f'.build/development-tuning-2026-10-11/floor{n}-runtime-probe-v1/result.json');entry['probe_sha256']=sha(entry['reuse_probe'])
            entries.append(entry)
    files=[Path(__file__),s.DECL,e.DECL,BASE/'.build/consistent_selected_development_v1.py',BASE/'.build/development_next_stage_v1.py',BASE/'.build/floor10_recipe_alias_review_v3.py',e.OUT/'recipe_export_alias_amendment_v3.json']
    for n in (9,10):files.extend([BASE/f'.build/development-tuning-2026-10-11/floor{n}-runtime-probe-v1/amendment.json',BASE/f'.build/development-tuning-2026-10-11/floor{n}-runtime-probe-v1/result.json'])
    save(DECL,dict(declared_at=datetime.datetime.now().astimezone().isoformat(),entries=entries,proofs=proofs,source_hashes={str(p):sha(p) for p in files},physics_changed=False,thresholds_changed=False,original_failures_preserved=True,holdout_used=False,population_analysis=False,maximum_attempts=1,scope='Terminal runtime guards only; future guards require a new amendment. Review uses original denominator and rolewise80.',observer_recovery='Original IO observer exited on transient reference hash mismatch; current original checked() passes all frozen references. No reference hashes changed.'))
    jobs=[]
    for x in entries:
        dest=ROOT/'cases'/x['id'];status=dest/'status.json'
        if x.get('reuse_probe'):
            r=read(x['reuse_probe']);r.update(runtime_completion_provenance=x,operational_acceptance=False)
            save(dest/'result.json',r);save(status,dict(state='complete',reused_completed_probe=True))
        else:jobs.append(dict(id=x['id'],kind='runtime_study',package='8-10 floor',entry='.build/runtime_completion_v1.py',arguments=['--action','case','--job',x['id']],status_path=str(status.relative_to(BASE)).replace('\\','/')))
    save(ROOT/'jobs.json',jobs);print(json.dumps(dict(entries=len(entries),new_jobs=len(jobs),proofs=proofs)))
def checked():
    d=read(DECL)
    for p,h in d['source_hashes'].items():assert sha(p)==h,p
    for x in d['entries']:assert sha(x['old_result'])==x['old_sha256']
    return d
def case(id):
    d=checked();x=next(x for x in d['entries'] if x['id']==id);dest=ROOT/'cases'/id
    if (dest/'status.json').exists():raise RuntimeError('Already attempted; no retry')
    save(dest/'status.json',dict(state='running',pid=os.getpid()))
    try:
        n=x['job']['stories'];controls=copy.deepcopy(s.checked() if n==9 else e.checked());controls['dynamic_runtime_s_per_direction']=x['runtime_s']
        if n==9:s.OUT=ROOT/'height9';s.install()
        else:e.OUT=ROOT/'height10'
        r=e.solve_dynamic(dict(x['job'],kind='development_dynamic'),controls);assert r['candidate_hash']==x['candidate_hash'];validate_result(r)
        r.update(job=x['job'],runtime_completion_provenance=x,holdout_used=False,population_analysis=False,scientific_qualification_passed=False)
        save(dest/'result.json',r);save(dest/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as exc:save(dest/'status.json',dict(state='needs_review',error=repr(exc)));raise

def report(n):
    d=checked();mod=s if n<10 else e;original_d=s.checked() if n<10 else e.checked()
    orig_result=mod.result;orig_terminal=mod.terminal;orig_save=mod.save
    entries={x['job']['id']:x for x in d['entries']}
    def result(j):
        x=entries.get(j['id'])
        if x:
            p=ROOT/'cases'/x['id']/'result.json';return read(p) if p.exists() else None
        return orig_result(j)
    def terminal(j):
        x=entries.get(j['id'])
        if x:
            p=ROOT/'cases'/x['id']/'status.json';return p.exists() and read(p)['state'] in ('complete','needs_review','failed')
        return orig_terminal(j)
    def annotated_save(p,v):
        if Path(p).name in ('accepted_recipe.json','development_review.json',f'{n}storey.json','floor10_development_review.json'):
            v=dict(v,runtime_completion_amendment=str(DECL),runtime_completion_amendment_sha256=sha(DECL),original_failures_preserved=True)
            if Path(p).name=='accepted_recipe.json':
                if n==10 and v['variant']=='reference_mesh':v.update(variant='mesh16',development_variant='reference_mesh',export_alias_amendment=str(ORIG[10]/'recipe_export_alias_amendment_v3.json'))
                v['source_hashes']={**v['source_hashes'],str(DECL):sha(DECL),str(Path(__file__)):sha(__file__)}
        orig_save(p,v)
    mod.result=result;mod.terminal=terminal;mod.save=annotated_save
    r=s.report(n,original_d) if n<10 else e.report_floor10(original_d)
    save(ROOT/f'reviews/{n}storey.json',dict(r,runtime_completion_amendment=str(DECL)))
def observe():
    while True:
        for n in range(6,11):
            py=BASE/('6-7 floor' if n<8 else '8-10 floor')/'runtime/python.exe'
            with (ROOT/f'review_{n}.stderr.log').open('a') as err:
                code=subprocess.run([str(py),'-X','utf8',str(Path(__file__)), '--action','report','--stories',str(n)],cwd=BASE,stderr=err).returncode
            if code:
                save(ROOT/'observer_status.json',dict(state='needs_review',stories=n,pid=os.getpid(),updated_at=time.time()));return
        save(ROOT/'observer_status.json',dict(state='running',pid=os.getpid(),updated_at=time.time(),numerical_solver=False))
        time.sleep(60)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['prepare','case','report','observe'],required=True);p.add_argument('--job');p.add_argument('--stories',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='case':case(a.job)
    elif a.action=='report':report(a.stories)
    else:observe()
