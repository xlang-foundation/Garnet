"""CPU-only session admission prefix checks; no native or GPU qualification."""
import ast,copy,json
from pathlib import Path
from types import SimpleNamespace
import unittest

class SessionWorkspaceTests(unittest.TestCase):
    def load(self):
        path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_speculative.py'
        cls=next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.ClassDef) and n.name=='CUDASpeculativeSession')
        method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
        stop=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='draft_builder_workspace_bytes' for t in n.targets))
        method.name='validate';method.body=method.body[:stop]+[ast.Return(value=ast.Constant(value=True))]
        module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]));namespace={'json':json}
        exec(compile(module,str(path),'exec'),namespace);return namespace['validate']
    def decoder(self,batch=4,capacity=4096,chunk=4):
        return SimpleNamespace(batch=batch,capacity=capacity,chunk_size=chunk,io=SimpleNamespace(memory_json=lambda:json.dumps({'vocab_size':100})))
    def test_256_profile_and_rejections(self):
        validate=self.load();self.assertTrue(validate(None,self.decoder(),'fixture',[],draft_builder_workspace_bytes=256<<20))
        for decoder,draft in [(self.decoder(batch=2),None),(self.decoder(batch=8),None),(self.decoder(capacity=512),None),(self.decoder(chunk=8),4),(self.decoder(),8),(self.decoder(batch=4.0),None)]:
            with self.assertRaises(ValueError):validate(None,decoder,'fixture',[],draft_chunk_size=draft,draft_builder_workspace_bytes=256<<20)
    def test_existing_defaults_and_invalid_budgets(self):
        validate=self.load()
        for budget in (0,128<<20):self.assertTrue(validate(None,self.decoder(batch=1,capacity=512),'fixture',[],draft_builder_workspace_bytes=budget))
        for budget in (True,None,64<<20,255<<20,512<<20,-1):
            with self.assertRaises(ValueError):validate(None,self.decoder(),'fixture',[],draft_builder_workspace_bytes=budget)

if __name__=='__main__':unittest.main()
