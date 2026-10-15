"""One Development8 pair with fresh process per direction after a native domain rebuild crash."""
import argparse,datetime,json,os,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'.build'))
import runtime_completion_v9 as prior
import runtime_completion_v1 as r
import consistent_selected_development_v1 as s
ROOT=r.ROOT.parent/'floor8-direction-isolation-v1';DECL=ROOT/'amendment.json';OLDID='selected8_dynamic_463eddaaa67e1d00bdd9a565'
read,save,sha=r.read,r.save,r.sha

def prepare():
 if DECL.exists():raise RuntimeError('Already frozen')
 d=s.checked();j=next(j for j in d['jobs'] if j['id']==OLDID);old=s.OUT/'cases'/OLDID;log=BASE/'.build'/f'{OLDID}.log'
 assert read(old/'status.json')['state']=='needs_review' and not (old/'result.json').exists()
 assert 'Windows fatal exception: access violation' in log.read_text(encoding='utf-8',errors='replace')
 files=[Path(__file__),s.DECL,prior.DECL,BASE/'.build/runtime_completion_v9.py',BASE/'.build/runtime_completion_v1.py',BASE/'.build/consistent_selected_development_v1.py',BASE/'.build/autonomous_development_v1.py',BASE/'.build/isolated_worker.py',BASE/'.build/run_floor8_refinement_study_v3.py',log,old/'status.json',old/'pre_exit_status.json']
 files += [BASE/'8-10 floor/src/fragility_poc'/name for name in ('fishbone_v5_batched_dynamic.py','fishbone_v4_dynamic.py','fishbone_v4_grouped_frame.py')]
 save(DECL,dict(declared_at=datetime.datetime.now().astimezone().isoformat(),job=j,candidate_hash=d['candidate_hashes']['8'],source_hashes={str(p):sha(p) for p in files},maximum_runtime_s_per_direction=7200,minimum_substep_factor=64,chunk_steps=10,maximum_attempts_per_direction=1,physics_changed=False,thresholds_changed=False,old_failure_preserved=True,holdout_used=False,population_analysis=False,reason='Native access violation stack ended in groupedmodel build after analyze activity, consistent with domain rebuild failure; exact causeunconfirmed. FreshX/Y independent solver processes, mathematical combination unchanged. No olddirectionprefix reused.',runtime_provenance='Outerselecteddecl says1200 but frozen floor8 runtime_point overrides to7200 in actualoldstack. Thisexplicitisolation retains7200effectivebudget; doesnotclaimold1200execution.'))
 jobs=[]
 for axis in ('X','Y'):
  id='isolated8_'+OLDID+'_'+axis
  jobs.append(dict(id=id,kind='runtime_study',package='8-10 floor',entry='.build/floor8_dynamic_direction_isolation_v1.py',arguments=['--action','unit','--job',id],status_path=str((ROOT/'units'/axis/'status.json').relative_to(BASE)).replace('\\','/')))
 save(ROOT/'jobs.json',jobs);print(json.dumps(dict(units=2,solver_launches=0)))
def checked():
 d=read(DECL)
 for p,h in d['source_hashes'].items():assert sha(p)==h,p
 return d
def unit(id):
 import numpy as np
 d=checked();axis=id.rsplit('_',1)[-1];assert axis in ('X','Y')
 out=ROOT/'units'/axis
 if (out/'status.json').exists():raise RuntimeError('Already attempted; inspect evidence, no retryloop')
 save(out/'status.json',dict(state='running',pid=os.getpid(),started_at=time.time()))
 try:
  s.checked();cw,c,candidate,a,b=s.inputs(d['job']);assert candidate['freeze_hash']==d['candidate_hash']
  from fragility_poc import fishbone_v5_batched_dynamic as native
  with cw.connect(c['database_path']) as conn:pair=dict(conn.execute('SELECT * FROM ground_motion_catalog WHERE pair_id=?',(d['job']['pair_id'],)).fetchone())
  tx,ax=native.load_acceleration(pair['component_x_path'],pair['units']);ty,ay=native.load_acceleration(pair['component_y_path'],pair['units']);assert np.allclose(tx,ty,rtol=0,atol=1e-10)
  sig=native.pair_case_signature(a,pair,d['job']['im_g'],candidate_definition_hash=candidate['candidate_model_definition_hash'],damping_ratio=float(c['model']['damping_ratio']),damping_model=str(c['model']['damping_model']),dynamic_instability_drift=float(c['model']['dynamic_instability_drift']),chunk_steps=10)
  v=native._run_direction_batched(a,b,c,axis,ax if axis=='X' else ay,record_dt_s=float(pair['dt_s']),scale_factor=float(sig['scale_factor']),damping_ratio=float(c['model']['damping_ratio']),damping_model=str(c['model']['damping_model']),dynamic_instability_drift=float(c['model']['dynamic_instability_drift']),maximum_runtime_s=7200,minimum_substep_factor=64,chunk_steps=10)
  arrays=v.pop('arrays');assert (np.diff(arrays['time_s'])>=0).all()
  for arr in arrays.values():assert np.isfinite(arr).all()
  np.savez_compressed(out/'history.npz',**arrays)
  save(out/'result.json',cw.json_values(dict(v,job=d['job'],direction=axis,candidate_hash=d['candidate_hash'],signature=sig,history_path=str(out/'history.npz'),history_sha256=sha(out/'history.npz'),amendment_sha256=sha(DECL),fresh_candidate_histories=True,physical_collapse_claim=False)))
  save(out/'status.json',dict(state='complete',pid=os.getpid(),result_status=v['status'],completed_at=time.time()))
 except BaseException as exc:save(out/'status.json',dict(state='needs_review',error=repr(exc),pid=os.getpid()));raise

def assemble():
 import numpy as np
 d=checked();dest=ROOT/'pair'
 if (dest/'result.json').exists():return
 if not all((ROOT/'units'/axis/'result.json').exists() for axis in ('X','Y')):return
 from fragility_poc.fishbone_v4_dynamic import _combine_directional_results
 vals={}
 for axis in ('X','Y'):
  v=read(ROOT/'units'/axis/'result.json');assert v['candidate_hash']==d['candidate_hash'] and v['job']==d['job'] and sha(v['history_path'])==v['history_sha256']
  with np.load(v['history_path']) as z:v['arrays']={k:z[k] for k in z.files}
  vals[axis]=v
 assert vals['X']['signature']==vals['Y']['signature']
 arrays,summary=_combine_directional_results(None,vals['X'],vals['Y'])
 for arr in arrays.values():assert np.isfinite(arr).all()
 dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'history.npz',**arrays)
 peak,proof=r.e.history_peak(dest/'history.npz');assert abs(peak-summary['max_midr'])<1e-10
 status='cp_endpoint_reached' if proof else summary['status'];failure=False if proof else summary['numerical_failure']
 result=dict(job=d['job'],candidate_hash=d['candidate_hash'],status=status,numerical_failure=failure,usable=not failure and status in ('success','cp_endpoint_reached','dynamic_instability'),candidate_midr=peak,history_path=str(dest/'history.npz'),history_sha256=sha(dest/'history.npz'),cp_endpoint_evidence=proof,original_directional_results={axis:{k:v for k,v in vals[axis].items() if k not in ('arrays','recovery_log')} for axis in vals},physical_collapse_claim=False,fresh_candidate_histories=True,original_pair_status=summary['status'],isolation_amendment=str(DECL),original_crash_log=str(BASE/'.build'/f'{OLDID}.log'),holdout_used=False,population_analysis=False,scientific_qualification_passed=False)
 save(dest/'result.json',result);save(dest/'status.json',dict(state='complete',result_status=status))
def report(n):
 d=checked();assemble();old_result=s.result;old_terminal=s.terminal;old_save=s.save
 def result(j):
  if j['id']==OLDID:
   p=ROOT/'pair/result.json';return read(p) if p.exists() else None
  return old_result(j)
 def terminal(j):return (ROOT/'pair/result.json').exists() if j['id']==OLDID else old_terminal(j)
 def annotated_save(p,v):
  if Path(p).name in ('accepted_recipe.json','development_review.json',f'{n}storey.json') and n==8:
   v=dict(v,direction_isolation_amendment=str(DECL),old_native_crash_preserved=True)
   if Path(p).name=='accepted_recipe.json':v['source_hashes']={**v['source_hashes'],str(DECL):sha(DECL),str(Path(__file__)):sha(__file__)}
  old_save(p,v)
 if n==8:s.result=result;s.terminal=terminal;s.save=annotated_save
 prior.report(n)
 save(ROOT/f'reviews/{n}storey.json',read(prior.ROOT/f'reviews/{n}storey.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',required=True,choices=['prepare','unit','report']);p.add_argument('--job');p.add_argument('--stories',type=int);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='unit':unit(a.job)
 else:report(a.stories)
