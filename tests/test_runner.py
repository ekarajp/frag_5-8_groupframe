import math
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from common import ROOT,read,THRESHOLDS,write
from policy import decide
from engine import Engine,DEFAULT_POLICY
from reports import fit_complete

def until(check,timeout=30):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if check():return
        time.sleep(.05)
    raise AssertionError('Timed out')

class PolicyTests(unittest.TestCase):
    def test_crossings_and_refinement(self):
        points=[]
        for _ in range(100):
            d=decide(points,DEFAULT_POLICY)
            if d['state']!='continue':break
            im=d['next_im_g']
            points.append({'target_im_g':im,'max_midr':im/10,'status':'success','numerical_failure':False})
        self.assertEqual(d['state'],'complete')
        for name,threshold in THRESHOLDS.items():
            c=d['capacities'][name]
            self.assertLess(abs(c['capacity_im_g']-10*threshold),1e-10)
            self.assertLessEqual(c['relative_bracket_width'],.05)
    def test_failure_is_never_CP(self):
        p=[{'target_im_g':.05,'max_midr':.001,'status':'success'},
           {'target_im_g':.1,'max_midr':.2,'status':'numerical_failure','numerical_failure':True}]
        d=decide(p,DEFAULT_POLICY)
        self.assertEqual(d['state'],'needs_review')
        self.assertFalse(d['capacities']['CP']['certified'])
    def test_initial_aboveIO_hunts_down(self):
        d=decide([{'target_im_g':.05,'max_midr':.02,'status':'success'}],DEFAULT_POLICY)
        self.assertLess(d['next_im_g'],.05)
    def test_guards_are_terminal_review(self):
        policy={**DEFAULT_POLICY,'minimum_im_g':.05}
        d=decide([{'target_im_g':.05,'max_midr':.06,'status':'cp_endpoint_reached'}],policy)
        self.assertEqual(d['state'],'needs_review')
    def test_fit_known_lognormal(self):
        f=fit_complete([math.exp(-.3),1,math.exp(.3)])
        self.assertAlmostEqual(f['theta_g'],1)
        self.assertAlmostEqual(f['beta_log_sample_ddof1'],.3)
    def test_changed_source_refused_before_analysis(self):
        import subprocess
        from common import digest
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)
            run={'policy':DEFAULT_POLICY,'code_hashes':[{'path':'app/worker.py','sha256':'0'*64}]}
            run['run_hash']=digest(run)
            write(out/'run.json',run);write(out/'job.json',{'id':'QA','kind':'modal'})
            p=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'app/worker.py'),
                '--job',str(out/'job.json'),'--run',str(out/'run.json')],capture_output=True,timeout=10)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Frozen scientific/application source changed',read(out/'error.json')['error'])
            self.assertFalse((out/'receipt.json').exists())
            self.assertFalse((out/'environment.json').exists())

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.e=Engine(self.temp.name,ROOT/'tests/fake_worker.py',minimum_free_gib=0,
                      catalog=read(ROOT/'assets/catalog.json')[:6])
        self.e.verified=True  # lifecycle test; full bundle verification tested separately
    def tearDown(self):
        self.e.stop()
        until(lambda:not self.e.processes)
        self.e.close()
        self.e.db.close()
        self.temp.cleanup()
    def test_continuous_refill_stop_resume_and_workers(self):
        self.e.start(3)
        until(lambda:len(self.e.processes)==3)
        self.e.set_workers(1)
        self.assertEqual(len(self.e.processes),3)  # no killing when target decreases
        self.e.stop()
        until(lambda:not self.e.processes)
        first=self.e.snapshot()['counts']['complete']
        self.e.start(2)
        until(lambda:self.e.snapshot()['models_terminal']==6)
        s=self.e.snapshot()
        self.assertEqual(s['models_certified'],6)
        self.assertEqual(s['overall_percent'],100.)
        self.assertGreater(s['counts']['complete'],first)
        self.assertEqual(s['counts']['needs_review'],0)
        self.assertEqual(self.e.db.execute('SELECT count(*) FROM jobs').fetchone()[0],42)
    def test_process_identity_not_pid_only(self):
        from common import identity
        import os
        i=identity(os.getpid())
        self.assertTrue(i['exe'])
        self.assertTrue(i['created'])
        self.assertIsNone(identity(2147483647))
    def test_reopen_adopts_live_workers_without_duplicate(self):
        self.e.start(2)
        until(lambda:len(self.e.processes)==2)
        run_dir=self.e.run_dir
        old_handles=[v['process'] for v in self.e.processes.values()]
        self.e.close()
        self.e.db.close()
        self.e=Engine(self.temp.name,ROOT/'tests/fake_worker.py',minimum_free_gib=0,
                      catalog=read(ROOT/'assets/catalog.json')[:6])
        self.e.verified=True
        until(lambda:not self.e.processes)
        for handle in old_handles:
            handle.wait(timeout=5)
        self.e.start(2)
        until(lambda:self.e.snapshot()['models_certified']==6)
        events=[__import__('json').loads(l) for l in (run_dir/'events.jsonl').read_text().splitlines()]
        modals=[r['job'] for r in events if r['event']=='dispatched' and r['job'].startswith('modal-')]
        self.assertEqual(len(modals),len(set(modals)))
    def test_failure_retained_other_models_continue(self):
        import json
        row=self.e.db.execute('SELECT id,payload FROM jobs LIMIT 1').fetchone()
        payload=json.loads(row['payload']);payload['simulate_failure']=True
        self.e.db.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(payload),row['id']))
        self.e.db.commit()
        self.e.start(3)
        until(lambda:self.e.snapshot()['models_terminal']==6)
        s=self.e.snapshot()
        self.assertEqual(s['models_certified'],5)
        self.assertEqual(s['models_review'],1)
        self.assertEqual(s['counts']['needs_review'],1)
        self.assertEqual(s['overall_percent'],100)
        self.assertLess(s['certified_percent'],100)

if __name__=='__main__':unittest.main(verbosity=2)
