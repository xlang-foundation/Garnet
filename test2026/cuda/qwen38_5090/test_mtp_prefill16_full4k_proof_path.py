"""CPU regression for proof bookkeeping; replay math/source guards unchanged."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,unittest
M=Path(__file__).resolve().parent


def tree(file):return ast.parse((M/file).read_bytes())
def target_assignment(t):return next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='target' for x in n.targets))


class ProofPathControls(unittest.TestCase):
    def test_only_proof_output_expression_changes_in_complete_verifier_ast(self):
        old=tree('verify_mtp_prefill16_full4k.py');new=tree('verify_mtp_prefill16_full4k_named.py')
        for t in (old,new):target_assignment(t).value=ast.Constant(value='proof-path-placeholder')
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
    def test_memory_loop_variable_cannot_choose_proof_filename(self):
        expression=ast.Expression(body=target_assignment(tree('verify_mtp_prefill16_full4k_named.py')).value);value=eval(compile(ast.fix_missing_locations(expression),'proof-path','eval'),{'args':SimpleNamespace(proofs=Path('synthetic-proofs')),'name':'memory_after'})
        self.assertEqual(value,Path('synthetic-proofs/mtp-prefill16-full4k-admission-attempt1-verified.json'))


if __name__=='__main__':unittest.main()
