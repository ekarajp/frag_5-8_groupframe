import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from engine import Engine
class FourHeightTests(unittest.TestCase):
    def test_ready_paused_with_two_thousand_models_and_variant_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(d)
            try:
                s=e.snapshot()
                self.assertFalse(s['running']);self.assertEqual(s['active'],[])
                self.assertEqual(s['models_total'],2000)
                self.assertEqual({f['stories']:f['total'] for f in s['floors']},{5:500,6:500,7:500,8:500})
                self.assertEqual(s['counts']['pending'],2000)
                self.assertTrue(any(r['path'].startswith('src_variants/5storey/') for r in e.run['code_hashes']))
                self.assertEqual(s['overall_percent'],0)
            finally:e.close();e.db.close()
        self.assertFalse(any(n.startswith('opensees') for n in sys.modules))
if __name__=='__main__':unittest.main(verbosity=2)
