"""Bounded runtime completion for two audited human-stop resumption guards only."""
import argparse,datetime,json,os,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'.build'))
import resume_selected_6_8_v1 as prior
import consistent_selected_development_v1 as s
ROOT=prior.ROOT.with_name('resume68-runtime-completion-v1');DECL=ROOT/'amendment.json'
read,save,sha=s.read,s.save,s.sha

def checked():
 d=read(DECL)
 for p,h in d['source_hashes'].items():assert sha(p)==h,p
 return d

def case(id):
 d=checked();x=next(x for x in d['entries'] if x['id']==id);j=x['job'];out=ROOT/'cases'/id
 if (out/'status.json').exists():raise RuntimeError('One bounded attempt already started; no automatic retry')
 save(out/'status.json',dict(state='running',pid=os.getpid(),started_at=time.time()))
 try:
  control=dict(s.checked());control['dynamic_runtime_s_per_direction']=7200;s.install();original_inputs=s.inputs
  def inputs(job):
   cw,c,candidate,a,b=original_inputs(job);c['run_dir']=str(ROOT/f'runs/{job["stories"]}storey');c['output_dir']=str(ROOT/f'outputs/{job["stories"]}storey');return cw,c,candidate,a,b
  s.engine.OUT=ROOT;s.engine.inputs=inputs
  v=s.engine.solve_dynamic(dict(j,kind='development_dynamic'),control)
  assert v['candidate_hash']==x['candidate_hash']
  v.update(job=j,runtime_completion_amendment=str(DECL),original_failed_result=x['old_result'],physical_collapse_claim=False,holdout_used=False,population_analysis=False,scientific_qualification_passed=False)
  save(out/'result.json',v);save(out/'status.json',dict(state='complete',pid=os.getpid(),completed_at=time.time(),result_status=v['status']))
 except BaseException as exc:save(out/'status.json',dict(state='needs_review',pid=os.getpid(),error=repr(exc)));raise

def report(n):
 d=checked();original=prior.read;oldsave=prior.save;paths={}
 for x in d['entries']:
  paths[str(Path(x['old_result']).resolve())]=ROOT/'cases'/x['id']/'result.json'
  paths[str(Path(x['old_result']).with_name('status.json').resolve())]=ROOT/'cases'/x['id']/'status.json'
 def overlay(p):
  target=paths.get(str(Path(p).resolve()))
  if target:return read(target) if target.exists() else ({'state':'pending'} if target.name=='status.json' else None)
  return original(p)
 def annotated(p,v):
  if Path(p).name in ('accepted_recipe.json','development_review.json',f'{n}storey.json'):
   v=dict(v,bounded_runtime_completion_amendment=str(DECL))
   if Path(p).name=='accepted_recipe.json':v['source_hashes']={**v['source_hashes'],str(DECL):sha(DECL),str(Path(__file__)):sha(__file__)}
  oldsave(p,v)
 prior.read=overlay;s.save=annotated;prior.save=annotated;prior.report(n)
 save(ROOT/f'reviews/{n}storey.json',read(prior.ROOT/f'reviews/{n}storey.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('case','report'),required=True);p.add_argument('--job');p.add_argument('--stories',type=int);a=p.parse_args()
 if a.action=='case':case(a.job)
 else:report(a.stories)

