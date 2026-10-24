"""CPU checks for diagnostic wrapper transparency, nesting and exception propagation."""
import ast,json,tempfile,time,unittest
from pathlib import Path
source=Path(__file__).with_name('probe_b4_mtp_state_wall_trace.py')
class TraceWrapperTests(unittest.TestCase):
 def test_wrappers(self):
  tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('patch','instrument')]
  with tempfile.TemporaryDirectory() as folder:
   barriers=[];scope={'originals':[],'active':False,'stack':[],'trace':Path(folder)/'trace.jsonl','time':time,'json':json,'barrier':lambda:barriers.append(1)}
   exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
   class Worker:
    def inner(self,value):return value+1
    def outer(self,value):return self.inner(value)*2
    def fail(self):raise RuntimeError('retained failure')
   original={name:getattr(Worker,name) for name in ('inner','outer','fail')}
   for name in original:scope['instrument'](Worker,name,name)
   try:
    w=Worker();self.assertEqual(w.outer(3),8);self.assertFalse(barriers);self.assertFalse(scope['trace'].exists())
    scope['active']=True;self.assertEqual(w.outer(3),8)
    with self.assertRaisesRegex(RuntimeError,'retained failure'):w.fail()
    rows=[json.loads(line) for line in scope['trace'].read_text().splitlines()]
    self.assertEqual([r['stage'] for r in rows],['inner','outer','fail'])
    self.assertEqual([(r['parent'],r['depth']) for r in rows],[('outer',1),(None,0),(None,0)])
    self.assertEqual(len(barriers),5);self.assertFalse(scope['stack'])
    self.assertTrue(all(r['ended']>=r['started'] and r['wall_seconds']>=0 for r in rows))
    self.assertIsNone(rows[0]['error']);self.assertIn('retained failure',rows[-1]['error'])
   finally:
    for owner,name,method in reversed(scope['originals']):setattr(owner,name,method)
   for name,method in original.items():self.assertIs(getattr(Worker,name),method)
if __name__=='__main__':unittest.main()
