"""CPU-only check of probe ordering and exact hook restoration, including errors."""
import ast
import json
import sys
import time
from pathlib import Path

source = Path(__file__).with_name('probe_qwen35_first_token.py')
tree = ast.parse(source.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'observed')
namespace = {'time': time, 'request_index': 0}


class Decoder:
    def chunk_hidden(self, hidden, flags, positions, predict=True):
        return [7, 99] if predict else None

    def prefill_sequence(self, *args):
        return self.chunk_hidden([], [], [], True)


class Speculative:
    def __init__(self, decoder, fail):
        self.decoder, self.fail = decoder, fail

    def _step(self, *args):
        if self.fail:
            raise RuntimeError('expected probe failure')
        return [8, 9]

    def prefill(self, *args):
        original = self.decoder.chunk_hidden
        existed = 'chunk_hidden' in self.decoder.__dict__
        saved = self.decoder.__dict__.get('chunk_hidden')
        self.decoder.chunk_hidden = lambda *a, **k: original(*a, **k)
        try:
            value = self.decoder.prefill_sequence([])
        finally:
            if existed:
                self.decoder.chunk_hidden = saved
            else:
                del self.decoder.chunk_hidden
        self._step(None, None, None, None, True, 'prefill')
        return value


class Model:
    batch = 2

    def __init__(self, fail):
        self.decoder = Decoder()
        self.speculative = Speculative(self.decoder, fail)

    def _prepare(self, *args):
        return None


def original_infer(model):
    for _ in range(model.batch):
        model._prepare(None, None)
    assert model.speculative.prefill([]) == [7, 99]
    return {'generated_token_ids': [[7], []], 'stop_token_ids': [None, 99]}


namespace['original_infer'] = original_infer
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
cases = []
for existing, fail in [(False, False), (True, False), (True, True)]:
    model = Model(fail)
    if existing:
        for obj, names in ((model, ('_prepare',)), (model.decoder, ('chunk_hidden', 'prefill_sequence')),
                           (model.speculative, ('_step', 'prefill'))):
            for name in names:
                setattr(obj, name, getattr(obj, name))
    owners = (model, model.decoder, model.speculative)
    before = [dict(obj.__dict__) for obj in owners]
    try:
        result = namespace['observed'](model)
        assert not fail
        assert result['first_token_boundary_diagnostic']['first_target_token_ids'] == [7, 99]
    except RuntimeError as error:
        assert fail and str(error) == 'expected probe failure'
    assert [obj.__dict__ for obj in owners] == before, 'Probe did not restore exact overrides'
    cases.append({'existing_overrides': existing, 'injected_failure': fail, 'passed': True})
output = Path(sys.argv[1])
assert not output.exists()
output.write_text(json.dumps({'all_hook_restoration_and_ordering_checks_passed': True, 'cases': cases,
    'scope': 'CPU fake protocol only; no CUDA/model/latency claim.'}, indent=2) + '\n')
print('QWEN35_FIRST_TOKEN_PROBE_CPU_HOOKS_PASS', len(cases))
