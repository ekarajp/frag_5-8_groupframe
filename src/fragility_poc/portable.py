"""Installation-independent identities and resumable project relocation."""
from pathlib import Path
import json, re, sqlite3
from functools import lru_cache
ROOT=Path(__file__).absolute().parents[2]

@lru_cache(maxsize=1)
def roots():
    p=ROOT/'.portable_roots.json'
    values=json.loads(p.read_text(encoding='utf-8')) if p.exists() else []
    return sorted(set(values+[str(ROOT)]),key=len,reverse=True)

def transform(value, replacement):
    if isinstance(value,str):
        if ':' not in value and not value.startswith('\\\\'): return value
        for base in roots():
            for variant in (base,base.replace('\\','/')):
                value=re.sub(re.escape(variant)+r'(?=$|[/\\])',lambda m:replacement,value,flags=re.I)
        return value
    if isinstance(value,dict): return {k:transform(v,replacement) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [transform(v,replacement) for v in value]
    return value

def canonical_paths(value): return transform(value,'__PROJECT_ROOT__')

def relocate():
    """Rebase only program-generated paths; content checksums stay unchanged."""
    p=ROOT/'.portable_roots.json'
    known=json.loads(p.read_text(encoding='utf-8')) if p.exists() else []
    location=ROOT/'.portable_location.json'
    if location.exists() and json.loads(location.read_text(encoding='utf-8')).get('root')==str(ROOT): return
    from .io_utils import atomic_write_json
    known=list(dict.fromkeys(known+[str(ROOT)]));atomic_write_json(p,known);roots.cache_clear()
    # A crash leaves the location marker untouched, so repeating is idempotent.
    if len(known)>1:
        for folder in ('data','runs','outputs'):
            for f in (ROOT/folder).rglob('*.json'):
                old=json.loads(f.read_text(encoding='utf-8'))
                new=transform(old,str(ROOT))
                if new!=old: atomic_write_json(f,new)
        for db in (ROOT/'data').rglob('*.sqlite'):
            with sqlite3.connect(db,timeout=60) as conn:
                tables=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                for table in tables:
                    qt='"'+table.replace('"','""')+'"'
                    columns=[r[1] for r in conn.execute(f'PRAGMA table_info({qt})') if str(r[2]).upper() in ('TEXT','VARCHAR')]
                    for col in columns:
                        qc='"'+col.replace('"','""')+'"'
                        for base in known:
                            if base==str(ROOT): continue
                            for old,new in [(base,str(ROOT)),(base.replace('\\','/'),str(ROOT).replace('\\','/')),(json.dumps(base)[1:-1],json.dumps(str(ROOT))[1:-1])]:
                                conn.execute(f'UPDATE {qt} SET {qc}=replace({qc},?,?) WHERE instr({qc},?)>0',(old,new,old))
    atomic_write_json(location,{'root':str(ROOT)})
