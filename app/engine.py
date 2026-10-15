"""Persistent work-conserving queue. No batches, native retries, or old results edits."""
import concurrent.futures
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from common import ROOT, read, write, digest, sha, identity, environment, verify_assets

DEFAULT_POLICY = {'name': 'portable-component-IDA-hunt-bisection-v2',
    'initial_im_g': .05, 'minimum_im_g': .001, 'maximum_im_g': 20.,
    'hunt_multiplier': 1.5, 'bracket_tolerance': .05, 'maximum_points_per_curve': 100,
    'maximum_scale_factor': 10000., 'runtime_s_per_direction': 7200.,
    'family_selection': 'nearest single CMS family; no midpoint blend', 'damping_ratio': .05,
    'numerical_failure_policy': 'retain, mark needs_review, no unchanged retry',
    'controller': 'deterministic hunt and log bisection; no surrogate/calibration',
    'recording': 'all committed single-component floor histories; native logs and recovery histories',
    'member_response': 'localForce and basicDeformation envelopes from every committed native step'}

class Engine:
    def __init__(self, state=None, worker_script=None, minimum_free_gib=10., catalog=None):
        self.state = Path(state or ROOT / 'state').resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.minimum_free_gib = minimum_free_gib
        self.worker_script = Path(worker_script or ROOT / 'app/worker.py')
        self.processes = {}
        self.active = False
        self.reason = 'Ready — no analyses started'
        self.closed = False
        self.verified = False
        self.preflight = {'state':'not_started','percent':0}
        self.workers = min(7, os.cpu_count() or 1)
        self.worker_limit = max(1, os.cpu_count() or 1)
        if (self.state/'settings.json').is_file():
            self.workers=max(1,min(self.worker_limit,int(read(self.state/'settings.json')['workers'])))
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.reporting = set()
        self.report_errors = {}
        self.final_export_submitted = False
        self.final_export_state = 'not_started'
        current = self.state / 'current.json'
        if current.exists():
            self.run_dir = self.state / read(current)['run']
            self.run = read(self.run_dir / 'run.json')
            if digest({k:v for k,v in self.run.items() if k!='run_hash'}) != self.run['run_hash']:
                raise ValueError('Frozen experiment plan changed')
            self.policy = self.run['policy']
            if self.run.get('schema')!='portable-component-fragility-campaign-v3': raise ValueError('Old paired state is not component state; migration required')
        else:
            self.run_dir = self.state / ('run-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
            self.run_dir.mkdir()
            self.policy = DEFAULT_POLICY.copy()
            plan_catalog = catalog if catalog is not None else read(ROOT/'assets/catalog.json')
            self.run = {'schema': 'portable-component-fragility-campaign-v3', 'policy': self.policy,
                'bundle_manifest_sha256': sha(ROOT / 'bundle_manifest.json'),
                'catalog_sha256': sha(ROOT / 'assets/catalog.json'),
                'motions_sha256': sha(ROOT / 'assets/motions.json'),
                'target_model_count':len(catalog) if catalog is not None else len(read(ROOT/'assets/catalog.json')),
                'height_target_counts':dict(Counter(str(r['stories']) for r in plan_catalog)),
                'created': time.time(), 'environment': environment(),
                'user_authorized_motions': 'installed CMS set; 23 available pairs, select by T1',
                'IM_definition':'Sa_component(T1_direction,5%)', 'component_results_independent':True, 'components_are_independent_events':False,
                'thresholds': {'IO':.01,'LS':.02,'CP':.04}, 'physical_collapse_claim': False}
            self.run['code_hashes'] = [r for r in read(ROOT/'bundle_manifest.json')['files']
                if r['path'].startswith(('app/','src/','src_variants/')) or r['path']=='assets/grouped_model_loader.py']
            self.run['run_hash'] = digest(self.run)
            write(self.run_dir / 'run.json', self.run)
            write(current, {'run': self.run_dir.name})
        if (self.run_dir/'research_inventory.json').is_file():
            self.final_export_state='complete'
            self.final_export_submitted=True
        self.db = sqlite3.connect(self.run_dir / 'queue.sqlite', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Queue SQLite integrity check failed')
        self.db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, model_id TEXT, kind TEXT,
            status TEXT, payload TEXT, pid_identity TEXT, started REAL, finished REAL, reason TEXT);
            CREATE TABLE IF NOT EXISTS model_reports(model_id TEXT PRIMARY KEY, state TEXT, error TEXT);
            CREATE INDEX IF NOT EXISTS job_status ON jobs(status,kind);
            CREATE INDEX IF NOT EXISTS job_model ON jobs(model_id);''')
        self.reporting = {r[0] for r in self.db.execute("SELECT model_id FROM model_reports WHERE state='complete'")
                          if (self.run_dir/'reports'/r[0]/'manifest.json').is_file()}
        self.catalog = catalog if catalog is not None else read(ROOT / 'assets/catalog.json')
        self.models = {f"{r['stories']}-{r['building_id']}":r for r in self.catalog}
        if not self.db.execute('SELECT count(*) FROM jobs').fetchone()[0]:
            for mid, model in self.models.items():
                self.add_job({'id': 'modal-' + mid, 'kind':'modal', 'model':model, 'model_id':mid})
            self.db.commit()
        # A reopened app never duplicates a live orphan native worker.
        for row in self.db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
            ident = json.loads(row['pid_identity'] or 'null')
            if ident and identity(ident['pid']) == ident:
                self.processes[row['id']] = {'process':None, 'identity':ident}
                self.event('adopted_live_worker',job=row['id'],identity=ident)
            else:
                self.finish(row, None)
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def event(self, name, **kw):
        with (self.run_dir / 'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'time':time.time(),'event':name,**kw},ensure_ascii=True)+'\n')
            f.flush()

    def add_job(self, payload):
        self.db.execute('INSERT OR IGNORE INTO jobs(id,model_id,kind,status,payload) VALUES(?,?,?,?,?)',
            (payload['id'],payload['model_id'],payload['kind'],'pending',json.dumps(payload)))

    def start(self, workers):
        if self.final_export_state in ('working','queued'):
            raise ValueError('Research inventory is being frozen; wait for export to finish')
        self.set_workers(workers)
        if sha(ROOT / 'bundle_manifest.json') != self.run['bundle_manifest_sha256']:
            raise ValueError('Bundle manifest changed since this campaign was frozen')
        if not self.verified:
            if self.preflight['state']=='verifying':
                raise ValueError('Frozen inputs are already being checked')
            self.preflight={'state':'verifying','percent':0}
            try:
                verify_assets(lambda p:setattr(self,'preflight',p))
            except Exception:
                self.preflight={'state':'failed','percent':0}
                raise
            self.preflight={'state':'passed','percent':100}
            self.verified = True
        with self.lock:
            if self.processes and (self.run_dir / 'stop.flag').exists():
                raise ValueError('Stop is draining a current point; wait for active workers to finish first')
            (self.run_dir / 'stop.flag').unlink(missing_ok=True)
            self.active = True
            self.reason = 'Running continuously'
            self.event('start',workers=self.workers)

    def stop(self):
        with self.lock:
            self.active = False
            write(self.run_dir / 'stop.flag', {'time':time.time()})
            self.reason = 'Stopping safely — finish current component point, then pause'
            self.event('stop_requested')

    def set_workers(self, count):
        if isinstance(count,bool) or int(count) != float(count) or not 1 <= int(count) <= self.worker_limit:
            raise ValueError(f'Workers must be 1..{self.worker_limit}')
        with self.lock:
            self.workers = int(count)
            write(self.state/'settings.json',{'workers':self.workers})
            self.event('workers_changed', workers=self.workers)

    def launch(self, row):
        job = json.loads(row['payload'])
        dest = self.run_dir / 'jobs' / row['id']
        dest.mkdir(parents=True,exist_ok=True)
        receipt = dest / 'receipt.json'
        if receipt.exists():
            old = read(receipt)
            if old['state'] != 'paused':
                raise ValueError('Terminal receipt exists for pending job; automatic retry refused')
            receipt.rename(dest / ('paused-receipt-' + str(time.time_ns()) + '.json'))
        write(dest / 'job.json', job)
        env = dict(os.environ)
        env.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        log = (dest / ('native-' + str(time.time_ns()) + '.log')).open('wb')
        try:
            p = subprocess.Popen([sys.executable, '-B', '-X','utf8',str(self.worker_script),
                '--job',str(dest / 'job.json'),'--run',str(self.run_dir / 'run.json')],
                cwd=ROOT, env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        finally:
            log.close()
        ident = identity(p.pid)
        self.processes[row['id']] = {'process':p,'identity':ident}
        self.db.execute("UPDATE jobs SET status='running',pid_identity=?,started=? WHERE id=?",
                        (json.dumps(ident),time.time(),row['id']))
        self.event('dispatched', job=row['id'], identity=ident)

    def finish(self, row, code):
        dest = self.run_dir / 'jobs' / row['id']
        state, reason = 'needs_review', 'Native process ended without a valid committed receipt'
        try:
            receipt = read(dest / 'receipt.json')
            if receipt['job_id'] != row['id'] or receipt['run_hash'] != self.run['run_hash']:
                raise ValueError('Receipt identity mismatch')
            if code not in (0,None):
                raise ValueError(f'Observed nonzero process exit {code}; receipt requires incident audit')
            state = receipt['state']
            reason = receipt.get('reason','')
            if state not in ('complete','paused','needs_review'):
                raise ValueError('Unknown receipt state')
            if row['kind'] == 'modal' and state == 'complete':
                if sha(dest / 'result.json') != receipt['result_sha256']:
                    raise ValueError('Modal result checksum mismatch')
                modal = read(dest / 'result.json')
                for component in modal['components']:
                    self.add_job({'id':row['model_id'] + '-' + component['component_id'],
                        'model_id':row['model_id'], 'kind':'curve','model':modal['model'],
                        **component})
            if row['kind'] == 'curve' and state == 'complete':
                if sha(dest/'curve.json') != receipt['curve_sha256']:
                    raise ValueError('Committed curve checksum mismatch')
                curve = read(dest / 'curve.json')
                if curve['state'] != 'complete' or not all(c['certified'] for c in curve['capacities'].values()):
                    raise ValueError('Curve does not have three certified brackets')
        except Exception as error:
            state, reason = 'needs_review', str(error)
        self.db.execute('UPDATE jobs SET status=?,reason=?,finished=? WHERE id=?',
                        ('pending' if state == 'paused' else state,reason,time.time(),row['id']))
        self.event('job_finished',job=row['id'],state=state,reason=reason,observed_exit=code)

    def report(self, mid, jobs):
        from reports import model_report
        try:
            model_report(self.run_dir,self.models[mid],jobs)
            with self.lock:
                self.db.execute('INSERT OR REPLACE INTO model_reports VALUES(?,?,?)',(mid,'complete',None))
                self.db.commit()
        except Exception as e:
            with self.lock:
                self.report_errors[mid] = str(e)
                self.db.execute('INSERT OR REPLACE INTO model_reports VALUES(?,?,?)',(mid,'needs_review',str(e)))
                self.db.commit()

    def finalize(self):
        from reports import campaign_export
        self.final_export_state = 'working'
        try:
            with self.lock:
                target = sqlite3.connect(self.run_dir/'queue_snapshot.sqlite')
                self.db.backup(target)
                target.close()
                write(self.run_dir/'job_ledger.json',[dict(r) for r in self.db.execute('SELECT * FROM jobs')])
            campaign_export(self.run_dir)
            self.final_export_state = 'complete'
            self.event('research_inventory_complete')
        except Exception as e:
            self.final_export_state = 'needs_review: '+str(e)
            self.event('research_inventory_error',error=str(e))
    def loop(self):
        while not self.closed:
            try:
                with self.lock:
                    for jid, item in list(self.processes.items()):
                        p = item['process']
                        if p is not None:
                            if p.poll() is None:
                                continue
                            code = p.returncode
                        else:
                            if identity(item['identity']['pid']) == item['identity']:
                                continue
                            code = None
                        row = self.db.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()
                        self.finish(row, code)
                        del self.processes[jid]
                    if self.active:
                        free = shutil.disk_usage(self.state).free/2**30
                        if free < self.minimum_free_gib:
                            self.stop()
                            self.reason = f'Disk reserve reached ({free:.1f} GiB); move the whole stopped package to a larger disk, then resume'
                        while self.active and len(self.processes) < self.workers:
                            row = self.db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY CASE kind WHEN 'curve' THEN 0 ELSE 1 END,rowid LIMIT 1").fetchone()
                            if not row:
                                break
                            self.launch(row)
                    # Fill every vacant native slot BEFORE scheduling independent report I/O.
                    terminal_models = self.db.execute('''SELECT model_id FROM jobs GROUP BY model_id
                        HAVING SUM(status IN ('pending','running'))=0''').fetchall()
                    for r in terminal_models:
                        mid = r[0]
                        if mid in self.reporting:
                            continue
                        self.reporting.add(mid)
                        jobs = [dict(r) for r in self.db.execute("SELECT * FROM jobs WHERE model_id=?",(mid,))]
                        self.pool.submit(self.report,mid,jobs)
                    self.db.commit()
                    if self.active and not self.processes and not self.db.execute("SELECT 1 FROM jobs WHERE status='pending' LIMIT 1").fetchone():
                        self.active = False
                        self.reason = 'All model jobs terminal; check certified / needs-review counts and report status'
                        self.event('queue_drained')
                        if not self.final_export_submitted:
                            self.final_export_submitted = True
                            self.pool.submit(self.finalize)
            except Exception as e:
                with self.lock:
                    self.active = False
                    self.reason = 'Controller needs review: ' + str(e)
                    write(self.run_dir / 'stop.flag', {'error':str(e)})
                    self.event('controller_error',error=str(e))
            time.sleep(.15)  # work-conserving refill, no batch polling delay

    def snapshot(self):
        with self.lock:
            rows = [dict(r) for r in self.db.execute('SELECT id,model_id,kind,status,reason FROM jobs')]
            counts = {s:sum(r['status']==s for r in rows) for s in ('pending','running','complete','needs_review')}
            floors, models = [], []
            report_bad = {r[0] for r in self.db.execute("SELECT model_id FROM model_reports WHERE state='needs_review'")}
            by_model = {}
            for row in rows:
                by_model.setdefault(row['model_id'],[]).append(row)
            for mid, model in self.models.items():
                jobs = by_model.get(mid,[])
                terminal = bool(jobs) and all(r['status'] in ('complete','needs_review') for r in jobs)
                good = terminal and all(r['status']=='complete' for r in jobs) and mid not in report_bad
                models.append({'id':mid,'stories':model['stories'],'building_id':model['building_id'],
                    'terminal':terminal,'certified':good, 'needs_review': terminal and not good,
                    'curves_complete':sum(r['kind']=='curve' and r['status']=='complete' for r in jobs),
                    'curves_expected':sum(r['kind']=='curve' for r in jobs)})
            for n in sorted({r['stories'] for r in self.catalog}):
                ms = [m for m in models if m['stories']==n]
                floors.append({'stories':n,'total':len(ms),'terminal':sum(m['terminal'] for m in ms),
                    'certified':sum(m['certified'] for m in ms),'review':sum(m['needs_review'] for m in ms)})
            active = []
            for jid,item in self.processes.items():
                a = self.run_dir / 'jobs' / jid / 'activity.json'
                try:
                    activity = read(a) if a.exists() else {}
                except (OSError,ValueError):
                    activity = {}
                active.append({'job':jid,'identity':item['identity'],'activity':activity})
            return {'running':self.active,'reason':self.reason,'workers':self.workers,'worker_limit':self.worker_limit,
                'active':active,'counts':counts,'floors':floors,'models_total':len(models),
                'models_terminal':sum(m['terminal'] for m in models),'models_certified':sum(m['certified'] for m in models),
                'models_review':sum(m['needs_review'] for m in models),
                'overall_percent':100*sum(m['terminal'] for m in models)/max(len(models),1),
                'certified_percent':100*sum(m['certified'] for m in models)/max(len(models),1),
                'disk_free_gib':shutil.disk_usage(self.state).free/2**30,
                'reports':dict(self.db.execute('SELECT state,count(*) FROM model_reports GROUP BY state').fetchall()),
                'failures':[r for r in rows if r['status']=='needs_review'][:100],
                'run_hash':self.run['run_hash'],'run_dir':str(self.run_dir),'policy':self.policy,
                'curve_total_is_final':all(any(r['kind']=='modal' and r['status'] in ('complete','needs_review') for r in by_model.get(mid,[])) for mid in self.models),
                'report_errors':self.report_errors,'final_export_state':self.final_export_state,'preflight':self.preflight,
                'modal_complete':sum(r['kind']=='modal' and r['status']=='complete' for r in rows),
                'curves_complete':sum(r['kind']=='curve' and r['status']=='complete' for r in rows),
                'curves_expected_so_far':sum(r['kind']=='curve' for r in rows)}

    def close(self):
        self.stop()
        self.closed = True
        self.thread.join(timeout=3)
        self.pool.shutdown(wait=True)
        with self.lock:
            self.db.commit()
