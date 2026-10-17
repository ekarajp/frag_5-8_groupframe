"""Recompute bundle_manifest.json rows from disk after intentional source edits; no solver imports or analysis."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'bundle_manifest.json'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    changed=[]
    for row in manifest['files']:
        p=ROOT/row['path']
        if not p.is_file():raise SystemExit(f'Manifest file missing on disk: {row["path"]}')
        size,digest=p.stat().st_size,sha(p)
        if (size,digest)!=(row['bytes'],row['sha256']):changed.append(row['path'])
        row['bytes'],row['sha256']=size,digest
    tmp=MANIFEST.with_name(MANIFEST.name+'.tmp-'+str(os.getpid()))
    # The shipped manifest is Windows-origin CRLF; keep the byte format stable.
    tmp.write_bytes((json.dumps(manifest,ensure_ascii=True,indent=2)+'\n').replace('\n','\r\n').encode('utf-8'))
    os.replace(tmp,MANIFEST)
    print(f'Manifest rows checked: {len(manifest["files"])}; updated: {len(changed)}')
    for path in changed:print('  updated',path)
    print('No analyses run.')
if __name__=='__main__':main()
