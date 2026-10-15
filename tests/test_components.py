import math,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from components import select_components
from common import sha
from reports import fit_complete
class ComponentTests(unittest.TestCase):
    def test_three_or_four_pairs_make_six_or_eight_individual_spectra(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);t=np.arange(301)*.01;motions=[]
            for family,count in ((1.0,4),(1.5,3),(3.0,4)):
                for i in range(count):
                    r=dict(pair_id=f'T{family}-{i}',source_set='CMS_ZONE5',conditioning_period_s=family,dt_s=.01,units='g',valid=1,physical_pair_hash=f'{family}-{i}')
                    for axis,factor in (('x',1),('y',2)):
                        p=root/f'{family}-{i}-{axis}.txt';np.savetxt(p,np.c_[t,.01*factor*np.sin(t*6)])
                        r[f'component_{axis}_path']=str(p);r[f'sha256_{axis}']=sha(p)
                    motions.append(r)
            for period,count in ((1.,8),(1.5,6),(3.,8)):
                modes={a:dict(modes=[dict(period_s=period)]) for a in ('X','Y')}
                with patch('components.resolve',lambda p:Path(p)):cs,selection=select_components(motions,modes)
                self.assertEqual(len(cs),count);self.assertEqual(selection['family_s'],period)
                self.assertEqual(len({c['component_id'] for c in cs}),count)
                self.assertAlmostEqual(cs[1]['unscaled_sa_g']/cs[0]['unscaled_sa_g'],2.)
                self.assertAlmostEqual((.1/cs[1]['unscaled_sa_g'])/(.1/cs[0]['unscaled_sa_g']),.5)
    def test_cluster_counts_are_pairs_not_components(self):
        f=fit_complete([.2,.3,.25,.35,.22,.32],['a','a','b','b','c','c'])
        self.assertEqual(f['component_count'],6);self.assertEqual(f['parent_pair_count'],3)
        with self.assertRaises(ValueError):fit_complete([1,2,3,4],['a','a','b','b'])
if __name__=='__main__':unittest.main(verbosity=2)
