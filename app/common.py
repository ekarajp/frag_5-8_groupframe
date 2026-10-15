"""Durable, relocatable records. No OpenSees import in the dashboard process."""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Never inherit another installation's runtime root after copying the package.
os.environ['FRAGILITY_POC_RUNTIME_ROOT'] = str(ROOT/'runtime_storage')
os.environ['FRAGILITY_ALLOW_SYSTEM_DRIVE'] = '1'  # operator-selected portable location
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'assets'))
THRESHOLDS = {'IO': .01, 'LS': .02, 'CP': .04}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(3):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(.1)

def resolve(relative):
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Path outside portable application')
    return p

def identity(pid):
    """PID + OS creation time + absolute executable; no stale-PID adoption."""
    if sys.platform != 'win32':
        import psutil
        try:
            p = psutil.Process(int(pid))
            with p.oneshot():
                if p.status() == psutil.STATUS_ZOMBIE:
                    return None
                return {'pid': int(pid), 'exe': str(Path(p.exe()).resolve()),
                        'created': p.create_time()}
        except (psutil.Error, OSError, ValueError):
            return None
    import ctypes
    from ctypes import wintypes as w
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    k.OpenProcess.restype = w.HANDLE
    k.CloseHandle.argtypes = [w.HANDLE]
    k.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
    k.QueryFullProcessImageNameW.argtypes = [w.HANDLE,w.DWORD,w.LPWSTR,ctypes.POINTER(w.DWORD)]
    k.GetExitCodeProcess.argtypes = [w.HANDLE,ctypes.POINTER(w.DWORD)]
    handle = k.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        exit_code = w.DWORD()
        if not k.GetExitCodeProcess(handle,ctypes.byref(exit_code)) or exit_code.value != 259:
            return None
        times = [w.FILETIME() for _ in range(4)]
        size = w.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if not k.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return None
        if not k.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return None
        return {'pid': int(pid), 'exe': str(Path(buf.value).resolve()),
                'created': (times[0].dwHighDateTime << 32) + times[0].dwLowDateTime}
    finally:
        k.CloseHandle(handle)

def environment():
    from importlib import metadata
    packages = {}
    for name in ('numpy','scipy','matplotlib','openseespy','openseespywin','openseespylinux','psutil'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    executable = Path(sys.executable).resolve()
    try:
        executable_label = executable.relative_to(ROOT).as_posix()
    except ValueError:
        executable_label = str(executable)
    return {'python': sys.version, 'platform': platform.platform(),
            'executable': executable_label, 'packages': packages,
            'cpu_count': os.cpu_count(), 'time_utc': time.time(), 'scientific_source_variant':os.environ.get('COMPONENT_SCIENTIFIC_VARIANT','dashboard-no-native')}

def verify_assets(progress=None):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    manifest = read(ROOT / 'bundle_manifest.json')
    bad = []
    total_bytes=sum(r['bytes'] for r in manifest['files'])
    checked_bytes=0
    def check(row):
        p=resolve(row['path'])
        return p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    # Parallel I/O only: every original file/size/SHA comparison remains required.
    # The coordinator alone invokes progress callbacks and updates their counters.
    with ThreadPoolExecutor(max_workers=8,thread_name_prefix='asset-io') as pool:
        pending={pool.submit(check,row):row for row in manifest['files']}
        for index,future in enumerate(as_completed(pending),1):
            row=pending[future]
            if not future.result():bad.append(row['path'])
            checked_bytes+=row['bytes']
            if progress:
                progress({'state':'verifying','percent':100*checked_bytes/max(total_bytes,1),
                          'files':index,'total_files':len(manifest['files'])})
    if bad:
        raise ValueError('Portable package changed: ' + ', '.join(bad[:12]))
    return {'files': len(manifest['files']), 'sha256': sha(ROOT / 'bundle_manifest.json')}


def configure_scientific_source(stories):
    """One isolated worker selects its original height-specific source snapshot."""
    if stories not in (5,6,7,8):raise ValueError('Only heights 5 through 8 belong to this package')
    if any(n=='fragility_poc' or n.startswith('fragility_poc.') for n in sys.modules):
        raise ValueError('Select the scientific snapshot before importing a scientific module')
    source=ROOT/'src_variants/5storey' if stories==5 else ROOT/'src'
    sys.path.insert(0,str(source))
    if stories==5:
        # The old workstation-only I/O helper inferred its root from source depth
        # and rejected C:. Preserve its file, but route its I/O function through
        # the existing portable helper with this installation's explicit root.
        # This adapter changes no model, solver, IM, damping or recovery code.
        import importlib.util
        spec=importlib.util.spec_from_file_location('_portable_runtime_io',ROOT/'src/fragility_poc/runtime_storage.py')
        portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)
        spec5=importlib.util.spec_from_file_location('fragility_poc.runtime_storage',source/'fragility_poc/runtime_storage.py')
        original=importlib.util.module_from_spec(spec5);spec5.loader.exec_module(original)
        def runtime_io(project_root=None,runtime_root=None):
            return portable.configure_runtime_storage(ROOT,ROOT/'runtime_storage')
        original.configure_runtime_storage=runtime_io
        sys.modules['fragility_poc.runtime_storage']=original
    os.environ['COMPONENT_SCIENTIFIC_VARIANT']='5storey-original' if stories==5 else '6-8-original'
    return source
