"""Explicit human-stop resumption for heights6–8; immutable original evidence."""
import argparse,datetime,json,os,sys,time,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'.build'))
import consistent_selected_development_v1 as s
import floor8_dynamic_direction_isolation_v1 as iso
ROOT=s.OUT.parent/'resume-selected-6-8-v1-2026-10-13';DECL=ROOT/'amendment.json'
read,save,sha=s.read,s.save,s.sha

def checked():
 d=read(DECL)
 for p,h in d['source_hashes'].items():
  if sha(p)!=h:raise RuntimeError('Frozen resume source changed: '+p)
 return d

def case(id):
 d=checked();x=next(x for x in d['interrupted'] if x['id']==id);j=x['job'];out=ROOT/'cases'/id
 if (out/'status.json').exists():raise RuntimeError('One declared attempt already started; inspect before further revision')
 save(out/'status.json',dict(state='running',pid=os.getpid(),started_at=time.time(),human_stop_resumption=True))
 try:
  prior=s.checked();s.install();original_inputs=s.inputs
  def inputs(job):
   cw,c,candidate,a,b=original_inputs(job);c['run_dir']=str(ROOT/f'runs/{job["stories"]}storey');c['output_dir']=str(ROOT/f'outputs/{job["stories"]}storey');return cw,c,candidate,a,b
  s.engine.OUT=ROOT;s.engine.inputs=inputs
  v=s.engine.solve_dynamic(dict(j,kind='development_dynamic'),prior)
  assert v['candidate_hash']==prior['candidate_hashes'][str(j['stories'])]
  v.update(job=j,human_stop_resumption_amendment=str(DECL),original_interrupted_status=x['old_status'],holdout_used=False,population_analysis=False,scientific_qualification_passed=False)
  save(out/'result.json',v);save(out/'status.json',dict(state='complete',pid=os.getpid(),result_status=v['status'],completed_at=time.time()))
 except BaseException as exc:save(out/'status.json',dict(state='needs_review',error=repr(exc),pid=os.getpid()));raise

def report(n):
 assert n in (6,7,8);d=checked();orig=s.result;terminal=s.terminal;oldsave=s.save;en={x['job']['id']:x for x in d['interrupted']}
 def result(j):
  if j['id'] in en:
   p=ROOT/'cases'/en[j['id']]['id']/'result.json';return read(p) if p.exists() else None
  return orig(j)
 def term(j):
  if j['id'] in en:
   p=ROOT/'cases'/en[j['id']]['id']/'status.json';return p.exists() and read(p).get('state') in ('complete','failed','needs_review')
  return terminal(j)
 def annotated(p,v):
  if Path(p).name in ('accepted_recipe.json','development_review.json',f'{n}storey.json'):
   v=dict(v,resumption_scope=[6,7,8],human_stop_resumption_amendment=str(DECL))
   if Path(p).name=='accepted_recipe.json':v['source_hashes']={**v['source_hashes'],str(DECL):sha(DECL),str(Path(__file__)):sha(__file__)}
  oldsave(p,v)
 s.result=result;s.terminal=term;s.save=annotated
 iso.report(n);save(ROOT/f'reviews/{n}storey.json',read(iso.ROOT/f'reviews/{n}storey.json'))

def observe(once=False):
 checked()
 while True:
  for n in (6,7,8):
   py=BASE/('6-7 floor' if n<8 else '8-10 floor')/'runtime/python.exe'
   for attempt in range(3):
    p=subprocess.run([str(py),'-X','utf8',str(Path(__file__)),'--action','report','--stories',str(n)],cwd=BASE,capture_output=True,text=True,encoding='utf-8')
    if not p.returncode:break
    with (ROOT/'preserved_report_errors.jsonl').open('a') as f:f.write(json.dumps(dict(stories=n,attempt=attempt+1,stderr=p.stderr,time=time.time()))+'\n')
    if 'Frozen selected-development source/reference changed:' not in p.stderr or attempt==2:
     save(ROOT/'observer_status.json',dict(state='needs_review',stories=n,error=p.stderr,pid=os.getpid()));raise RuntimeError(p.stderr)
    time.sleep(2)
  rows=[read(ROOT/f'reviews/{n}storey.json') for n in (6,7,8)]
  save(ROOT/'observer_status.json',dict(state='complete' if all(x['development_complete'] for x in rows) else 'running',heights=[dict(stories=x['stories'],state=x['state'],coverage=x['coverage']) for x in rows],pid=os.getpid(),updated_at=time.time(),numerical_solver=False,population_analysis=False))
  if once or all(x['development_complete'] for x in rows):return
  time.sleep(60)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('case','report','observe','once'),required=True);p.add_argument('--job');p.add_argument('--stories',type=int);a=p.parse_args()
 if a.action=='case':case(a.job)
 elif a.action=='report':report(a.stories)
 else:observe(a.action=='once')
