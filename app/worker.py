"""One isolated native process per modal job or IDA curve; durable point resume."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import math
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace as NS
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read, write, sha, digest, resolve, environment

def save_history(path, arrays):
    import numpy as np
    path = Path(path)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary,path)

def validate_history(arrays, stories):
    import numpy as np
    expected = {'time_s': (), 'floor_displacement_m': (stories,),
                'story_deformation_m': (stories,), 'story_drift_ratio': (stories,),
                'floor_absolute_acceleration_m_s2': (stories,), 'base_shear_kn': ()}
    if set(arrays) != set(expected):
        raise ValueError('History array schema differs')
    t = arrays['time_s']
    if len(t) < 2 or t[0] != 0 or not np.all(np.diff(t) > 0):
        raise ValueError('Missing recorder output / non-increasing time')
    for key, suffix in expected.items():
        if arrays[key].shape != (len(t),) + suffix or not np.isfinite(arrays[key]).all():
            raise ValueError('Invalid history: ' + key)
    return float(np.max(np.abs(arrays['story_drift_ratio'])))

def setup(job, run):
    import grouped_model_loader as loader
    from fragility_poc.structural import _ops
    modelpath = resolve(job['model']['path'])
    model = loader.read_model(modelpath, job['model']['sha256'])
    recipe = read(resolve(job['model']['recipe']))
    if sha(resolve(job['model']['recipe'])) != model['recipe_sha256']:
        raise ValueError('Recipe identity mismatch')
    artifact = NS(**model['construction_inputs'])
    def build(*args, **kwargs):
        direction = args[3] if len(args) > 3 else kwargs['direction']
        info, audit = loader.build_model(modelpath, direction, _ops(), job['model']['sha256'])
        return NS(**info), NS(**audit)
    return model, recipe, artifact, build

def modal(job, run, out):
    from fragility_poc.fishbone_v4_grouped_frame import run_grouped_frame_gravity, grouped_frame_directional_modes
    from components import select_components
    model, recipe, a, build = setup(job, run)
    modes = {}
    for direction in ('X','Y'):
        info, audit = build(None, None, None, direction)
        error = run_grouped_frame_gravity(info, audit)
        if error > .01: raise ValueError('Gravity equilibrium error exceeds 1%')
        modes[direction] = grouped_frame_directional_modes(a, info, direction, requested_modes=a.story_count)
        modes[direction]['gravity_equilibrium_relative_error'] = error
    components, selection = select_components(read(ROOT/'assets/motions.json'), modes)
    result = {'kind':'modal','model':job['model'],'modes':modes,'components':components,
              'selection':selection,'candidate_hash':recipe['candidate_hash'],'environment':environment()}
    write(out/'result.json',result)
    return {'state':'complete','result_sha256':sha(out/'result.json')}


def curve(job, run, out):
    import numpy as np
    from fragility_poc import fishbone_v5_batched_dynamic as dyn
    from fragility_poc.ground_motion import load_acceleration
    from fragility_poc.structural import _ops
    from policy import decide
    model, recipe, artifact, build = setup(job, run)
    dyn.build_grouped_frame_model = build  # replay exact portable commands, never rebuild/recalibrate
    motion = next(p for p in read(ROOT / 'assets/motions.json') if p['pair_id'] == job['pair_id'])
    axis=job['axis']
    p=resolve(motion[f'component_{axis.lower()}_path'])
    if sha(p)!=motion[f'sha256_{axis.lower()}']: raise ValueError('Component hash changed')
    t,acc=load_acceleration(p,motion['units'])
    accelerations={axis:(t,acc)}
    # Each native process works in a private local scratch directory. Native paths
    # are ASCII filenames relative to cwd even when the portable root contains Thai.
    scratch = out / 'native_scratch'
    if scratch.exists():
        raise ValueError('Native scratch from an unfinished attempt exists; audit required')
    scratch.mkdir()
    previous_cwd = Path.cwd()
    os.chdir(scratch)
    original_temp = tempfile.TemporaryDirectory
    class RetainedRecorderDirectory:
        def __init__(self,*args,**kwargs):
            self.path = tempfile.mkdtemp(prefix='recorder_',dir=scratch)
        def __enter__(self):
            return self.path
        def __exit__(self,*args):
            pass  # removed only after validated directional NPZ commits
    def local_temp(*args, **kwargs):
        return RetainedRecorderDirectory()
    dyn.tempfile.TemporaryDirectory = local_temp
    ops = _ops()
    original_recorder = ops.recorder
    def ascii_recorder(*args):
        args=list(args)
        if '-file' in args:
            ix=args.index('-file')+1
            path=Path(args[ix]).absolute()
            args[ix]=os.path.relpath(path,scratch)
            if not args[ix].isascii():raise ValueError('Native scratch filename must be ASCII')
        return original_recorder(*args)
    ops.recorder=ascii_recorder
    original_analyze = ops.analyze
    original_analysis = ops.analysis
    heartbeat = {'last': 0.0, 'axis': None, 'im': None, 'stage': 'starting'}
    member = {'active':False}
    def analysis(*args):
        heartbeat['stage'] = 'NLTHA' if args[0]=='Transient' else 'gravity'
        result = original_analysis(*args)
        if args[0] == 'Transient' and not member['active']:
            member['active'] = True
            tags = list(ops.getEleTags())
            member['tags'] = tags
            member['groups'] = {}
            for response in ('localForce','basicDeformation'):
                groups = {}
                for tag in tags:
                    queries = (response,) if response=='localForce' else ('basicDeformation','basicDeformations')
                    for query in queries:
                        count = len(ops.eleResponse(tag,query))
                        if count:
                            groups.setdefault(query,[]).append((tag,count))
                            break
                    else:
                        raise ValueError(f'Missing member response {response} for element {tag}')
                member['groups'][response] = []
                for i,(query,entries) in enumerate(groups.items()):
                    filename = f'{response}-{i}.out'
                    member['groups'][response].append({'file':filename,'query':query,'entries':entries})
                    ops.recorder('Element','-file',str(scratch/filename),
                        '-precision',16,'-time','-ele',*[r[0] for r in entries],query)
        return result
    ops.analysis = analysis
    def analyze(*args):
        code = original_analyze(*args)
        now = time.monotonic()
        if now - heartbeat['last'] >= 1:
            heartbeat['last'] = now
            try:
                write(out / 'activity.json', {'pid': os.getpid(), 'time': time.time(),
                    'stage': heartbeat['stage'], 'axis': heartbeat['axis'], 'im_g': heartbeat['im'],
                    'analysis_time_s': float(ops.getTime()), 'duration_s': float(accelerations[axis][0][-1]),
                    'point_count': len(list(out.glob('point-*/component.json')))})
            except OSError:
                pass  # telemetry must never interrupt the native numerical method
        return code
    ops.analyze = analyze
    from legacy_seeds import import_seeds
    import_seeds(job,run,out,recipe)
    points = []
    # Only complete, hash-verified pair points can be resumed; no partial-prefix reuse.
    for p in sorted(out.glob('point-*/component.json')):
        r = read(p)
        for name, checksum in r['file_hashes'].items():
            if sha(p.parent / name) != checksum:
                raise ValueError('Saved point hash mismatch')
        if r['run_hash'] != run['run_hash']:
            raise ValueError('Saved point belongs to another plan')
        if r['component_id']!=job['component_id'] or r['axis']!=axis or r['model_sha256']!=job['model']['sha256'] or r['candidate_hash']!=recipe['candidate_hash']:
            raise ValueError('Saved point belongs to a different model/component/candidate')
        if r['gm_hashes']!=[motion[f'sha256_{axis.lower()}']] or not math.isclose(r['unscaled_sa_g'],job['unscaled_sa_g'],rel_tol=1e-12):
            raise ValueError('Saved point input/IM mismatch')
        if not math.isclose(r['scale_factor']*r['unscaled_sa_g'],r['target_im_g'],rel_tol=1e-10):
            raise ValueError('Saved point scaling mismatch')
        points.append(r)
    try:
        while True:
            decision = decide(points, run['policy'])
            write(out / 'curve.json', {'model': job['model'], 'pair_id': job['pair_id'],
                'reference_period_s': job['reference_period_s'], 'component_id':job['component_id'], 'axis':job['axis'], 'run_hash': run['run_hash'],
                'points': points, **decision})
            if decision['state'] != 'continue':
                return decision
            if (out.parents[1] / 'stop.flag').exists():
                return {'state': 'paused'}
            im = decision['next_im_g']
            sf = im / job['unscaled_sa_g']
            if sf > run['policy']['maximum_scale_factor']:
                return {'state': 'needs_review', 'reason': 'scale_factor_guard'}
            pointdir = out / f'point-{len(points):03d}'
            if pointdir.exists():
                raise ValueError('Uncommitted point exists; audit required, automatic retry refused')
            pointdir.mkdir()
            directional = {}
            hashes = {}
            for axis in (job['axis'],):
                member['active'] = False
                heartbeat.update(axis=axis, im=im, stage='gravity_modal_transient')
                result = dyn._run_direction_batched(artifact, model['building'],
                    {'model': recipe['model_config'], 'building': recipe['building_config']},
                    axis, accelerations[axis][1], record_dt_s=float(motion['dt_s']),
                    scale_factor=sf, damping_ratio=.05, damping_model='modal',
                    dynamic_instability_drift=.04,
                    maximum_runtime_s=run['policy']['runtime_s_per_direction'],
                    minimum_substep_factor=64, chunk_steps=10)
                peak = validate_history(result['arrays'], artifact.story_count)
                if result['gravity_equilibrium_relative_error'] > .01:
                    raise ValueError('Gravity equilibrium error exceeds 1%')
                result['committed_cp_proof'] = bool(peak >= .04)
                result['peak_midr_raw'] = peak
                # Native flags are preserved; research interpretation is separate.
                result['physical_collapse_claim'] = False
                hp = pointdir / (axis + '.npz')
                save_history(hp, result['arrays'])
                hashes[hp.name] = sha(hp)
                # Stream raw member recorder rows into envelopes, avoiding a giant
                # in-memory member-history matrix in each native worker.
                envelope = {'element_tags':np.asarray(member['tags'],dtype=np.int64)}
                for response in ('localForce','basicDeformation'):
                    minima,maxima,counts,tags,sample_counts = [],[],[],[],[]
                    for group in member['groups'][response]:
                        width = sum(r[1] for r in group['entries'])
                        lo, hi = np.full(width,np.inf), np.full(width,-np.inf)
                        samples, last = 0, -1.
                        with (scratch/group['file']).open('r',encoding='ascii') as f:
                            for line in f:
                                values = np.fromstring(line,sep=' ')
                                if len(values) != width+1 or not np.isfinite(values).all() or values[0] <= last:
                                    raise ValueError('Invalid member recorder: ' + response)
                                last = values[0]
                                if last <= result['termination_time_s']+1e-9:
                                    lo = np.minimum(lo,values[1:]); hi = np.maximum(hi,values[1:]); samples += 1
                        if not samples:
                            raise ValueError('No member response samples captured')
                        minima.append(lo);maxima.append(hi);sample_counts.append(samples)
                        tags.extend(r[0] for r in group['entries']);counts.extend(r[1] for r in group['entries'])
                    if len(set(sample_counts)) != 1:
                        raise ValueError('Member recorder sample counts differ')
                    lo,hi = np.concatenate(minima),np.concatenate(maxima)
                    envelope[response+'_element_tags'] = np.asarray(tags,dtype=np.int64)
                    envelope[response+'_min'],envelope[response+'_max'] = lo,hi
                    envelope[response+'_absmax'] = np.maximum(abs(lo),abs(hi))
                    envelope[response+'_offsets'] = np.cumsum([0]+counts)
                    envelope[response+'_samples'] = np.asarray(sample_counts[0])
                ep = pointdir/(axis+'-members.npz')
                save_history(ep,envelope)
                hashes[ep.name] = sha(ep)
                write(pointdir / (axis + '.json'), {k:v for k,v in result.items() if k != 'arrays'})
                hashes[axis + '.json'] = sha(pointdir / (axis + '.json'))
                directional[axis] = result
                for raw in scratch.iterdir():
                    if raw.is_dir():
                        shutil.rmtree(raw)
                    else:
                        raw.unlink()
            unit=directional[axis]
            summary={k:v for k,v in unit.items() if k!='arrays'}
            peak=unit['peak_midr_raw']
            valid=not unit['numerical_failure'] and (unit['completed_fraction']>=1-1e-8 or unit['committed_cp_proof'])
            status='numerical_failure' if not valid else ('cp_endpoint_reached' if peak>=.04 else 'success')
            point={**summary,'native_directional_status':unit['status'],'status':status,
                'schema':'portable-component-IDA-point-v2','component_id':job['component_id'],
                'pair_id':job['pair_id'],'axis':axis,'max_midr':peak,
                'numerical_failure':not valid,'physical_collapse_claim':False,
                'target_im_g':im,'scale_factor':sf,'unscaled_sa_g':job['unscaled_sa_g'],
                'IM_definition':'Sa_component(T1_direction,5%)',
                'reference_period_s':job['reference_period_s'],'file_hashes':hashes,
                'run_hash':run['run_hash'],'model_sha256':job['model']['sha256'],
                'candidate_hash':recipe['candidate_hash'],'gm_hashes':[motion[f'sha256_{axis.lower()}']],
                'output_sampling':'every_committed_step','point_index':len(points)}
            point['case_signature'] = digest({k:point[k] for k in
                ('run_hash','target_im_g','scale_factor','reference_period_s','model_sha256','candidate_hash','gm_hashes')})
            write(pointdir / 'component.json', point)
            points.append(point)
    finally:
        os.chdir(previous_cwd)
        if not any(scratch.iterdir()):
            scratch.rmdir()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    jobpath = Path(args.job).resolve()
    out = jobpath.parent
    try:
        job, run = read(jobpath), read(args.run)
        if digest({k:v for k,v in run.items() if k != 'run_hash'}) != run['run_hash']:
            raise ValueError('Run declaration hash mismatch')
        for row in run.get('code_hashes',[]):
            if sha(resolve(row['path'])) != row['sha256']:
                raise ValueError('Frozen scientific/application source changed: '+row['path'])
        from common import configure_scientific_source
        configure_scientific_source(job['model']['stories'])
        write(out / 'environment.json', environment())
        result = modal(job, run, out) if job['kind'] == 'modal' else curve(job, run, out)
        # A receipt is written only after all original outputs and validation commit.
        if job['kind'] == 'curve' and (out/'curve.json').exists():
            result['curve_sha256'] = sha(out/'curve.json')
        write(out / 'receipt.json', {**result, 'job_id': job['id'], 'run_hash': run['run_hash'],
            'finished': time.time(), 'pid': os.getpid()})
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        write(out / 'error.json', {'error': error, 'time': time.time(), 'pid': os.getpid()})
        os._exit(1)

if __name__ == '__main__':
    main()
