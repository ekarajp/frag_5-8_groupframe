"""Cross-platform entry point. Never starts analyses automatically."""
import os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(ROOT/'app'))
if __name__=='__main__':
    runpy.run_path(str(ROOT/'app/server.py'),run_name='__main__')
