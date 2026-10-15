"""Download/extract immutable floor-5 seeds; no solver imports or analysis."""
import argparse,hashlib,json,os,shutil,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def install(archive,manifest,root):
    assert archive.stat().st_size==manifest['bytes'],'Archive size mismatch'
    assert sha(archive)==manifest['sha256'],'Archive SHA-256 mismatch'
    expected={r['path']:r for r in manifest['files']}
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==len(expected) and set(z.namelist())==set(expected),'Unexpected ZIP members'
        for entry in z.infolist():
            name=entry.filename
            assert name.startswith('assets/legacy_seeds/') and '\\' not in name
            dest=(root/name).resolve()
            assert dest.is_relative_to((root/'assets/legacy_seeds').resolve())
            row=expected[name];assert entry.file_size==row['bytes']
            if dest.exists():
                assert dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256'],f'Existing data differs: {name}'
                continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_name(dest.name+'.download-'+str(os.getpid()))
            try:
                with z.open(entry) as source,tmp.open('xb') as target:shutil.copyfileobj(source,target,4*1024*1024)
                assert sha(tmp)==row['sha256'],name
                os.replace(tmp,dest)
            finally:
                if tmp.exists():tmp.unlink()
    return len(expected)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);args=p.parse_args()
    m=read(ROOT/'distribution/legacy_seed_data.json')
    if all((ROOT/r['path']).is_file() and sha(ROOT/r['path'])==r['sha256'] for r in m['files']):
        print('All floor-5 seed files already verified. No analyses run.');return
    archive=args.archive
    if archive is None:
        cache=ROOT/'.downloads';cache.mkdir(exist_ok=True);archive=cache/m['name']
        if not archive.exists():
            tmp=archive.with_suffix('.download')
            print('Downloading immutable floor-5 reuse data:',m['url'],flush=True)
            with urllib.request.urlopen(m['url'],timeout=120) as source,tmp.open('wb') as target:shutil.copyfileobj(source,target,4*1024*1024)
            os.replace(tmp,archive)
    count=install(archive,m,ROOT)
    print(f'Verified {count} floor-5 files. No analyses run.')
if __name__=='__main__':main()
