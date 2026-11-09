"""Independent retained-buffer and FP64 audit. Starts no Metal work or model inference."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

s = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
r = s / 'correctness1'
report = json.loads((r / 'report1.json').read_text())
status = json.loads((s / 'run-status1.json').read_text())
assert status['status'] == 'complete' and status['returncode'] == 0
assert report['passed'] and not report['timing_collected'] and not report['native_integration']
assert len(report['cases']) == 5
conditions = json.loads((s / 'conditions1.json').read_text())
for name, expected in conditions['hashes'].items():
    assert hashlib.sha256((s / name).read_bytes()).hexdigest() == expected, name
guard, poison = 16, 0x7fc12345
exact = guards = oracle_values = zero_values = rejected = 0
max_fraction = 0.0
cases = []

def read(name, count):
    path = r / name
    assert path.stat().st_size == count * 4, name
    return np.memmap(path, mode='r', dtype='<f4', shape=(count,))

for ci, cell in enumerate(report['cases']):
    b, t, cap, extent, first = (cell[k] for k in ['batch','chunk','capacity','extent','first'])
    assert (b,t,cap,first,extent) == [(1,8,128,0,128),(4,17,256,128,256),
                                    (4,128,4096,1920,2048),(4,128,4096,3840,4096),
                                    (4,128,4096,1920,2048)][ci]
    queries, cp, lp = b*32*t, cap+4, t+4
    cw, lw = 2*guard+queries*cp, 2*guard+queries*lp
    source = read(f'case{ci}-input-cache.f32', cw)
    live = read(f'case{ci}-input-live.f32', lw)
    slots = read(f'case{ci}-slots.f32', 2*guard+b*(t+3))[guard:-guard].reshape(b,t+3)
    active = read(f'case{ci}-active.f32', 2*guard+b*(t+5))[guard:-guard].reshape(b,t+5)
    control = read(f'case{ci}-control-cache.f32', cw)
    control_live = read(f'case{ci}-control-live.f32', lw)
    candidate = read(f'case{ci}-candidate-cache.f32', cw)
    candidate_live = read(f'case{ci}-candidate-live.f32', lw)
    assert np.array_equal(control.view('<u4'), candidate.view('<u4'))
    assert np.array_equal(control_live.view('<u4'), candidate_live.view('<u4'))
    exact += cw+lw
    for payload, expected in [(control,source),(candidate,source)]:
        assert np.array_equal(payload[:guard].view('<u4'),expected[:guard].view('<u4'))
        assert np.array_equal(payload[-guard:].view('<u4'),expected[-guard:].view('<u4'))
        matrix = payload[guard:-guard].reshape(queries,cp)
        src = expected[guard:-guard].reshape(queries,cp)
        assert np.array_equal(matrix[:,extent:].view('<u4'),src[:,extent:].view('<u4'))
        assert np.isfinite(matrix[:,:extent]).all()
        guards += 2*guard+queries*(cp-extent)
    for payload in [control_live,candidate_live]:
        assert (payload[:guard].view('<u4') == poison).all()
        assert (payload[-guard:].view('<u4') == poison).all()
        matrix = payload[guard:-guard].reshape(queries,lp)
        assert (matrix[:,t:].view('<u4') == poison).all()
        assert np.isfinite(matrix[:,:t]).all()
        guards += 2*guard+queries*(lp-t)
    original = source[guard:-guard].reshape(queries,cp)
    live_scores = live[guard:-guard].reshape(queries,lp)
    out = candidate[guard:-guard].reshape(queries,cp)
    out_live = candidate_live[guard:-guard].reshape(queries,lp)
    for row, count in enumerate(cell['active_counts']):
        assert count <= t and np.array_equal(active[row,:t], np.arange(t)<count)
        assert np.array_equal(slots[row,:count], first+np.arange(count))
        for head in range(32):
            for token in range(t):
                q = (row*32+head)*t+token
                if token >= count:
                    assert not np.count_nonzero(out[q,:extent]) and not np.count_nonzero(out_live[q,:t])
                    zero_values += extent+t
        for head in [0,15,31]:
            for token in sorted(set([0,count//2,count-1])) if count else []:
                q = (row*32+head)*t+token
                visible = first+token+1
                logits = np.concatenate([original[q,:first],live_scores[q,:token+1]]).astype(np.float64)/math.sqrt(128)
                exp = np.exp(logits-logits.max())
                expected = exp/exp.sum(dtype=np.float64)
                got = np.concatenate([out[q,:first],out_live[q,:token+1]]).astype(np.float64)
                bound = 1e-6+2e-6*np.abs(expected)
                fraction = np.max(np.abs(got-expected)/bound)
                assert fraction <= 1
                max_fraction = max(max_fraction,float(fraction))
                assert abs(got.sum()-1) <= 2e-6
                assert not np.count_nonzero(out[q,first:extent])
                assert not np.count_nonzero(out_live[q,token+1:t])
                oracle_values += visible
    logical = queries*(extent+t)
    one_guards = 4*guard+queries*(cp-extent+lp-t)
    assert cell['rejected_control_variants'] == 5
    rejected += 5
    for v in cell['variants']:
        assert v['logical_words'] == logical and v['guard_words'] == one_guards
        assert v['threads'] == 128 and v['exact_to_control']
        assert v['dynamic_threadgroup_bytes'] == extent*(4 if v['variant']=='candidate' else 6)
    cases.append(dict(case=ci,batch=b,actual_chunk=t,context_bound=cap,extent=extent,
                      exact_parent_words=cw+lw,logical_words_per_variant=logical))

# The equality gate must notice a one-bit probability change, including NaN payloads.
def equality_gate(a,b):
    assert np.array_equal(a.view('<u4'),b.view('<u4'))
for original in [np.array([0.125,0.25],dtype='<f4'),np.array([poison],dtype='<u4').view('<f4')]:
    corrupt = original.copy()
    corrupt.view('<u4')[0] ^= 1
    try:
        equality_gate(original,corrupt)
    except AssertionError:
        continue
    raise AssertionError('Corrupted output/guard accepted')
result = dict(passed=True,scope='Retained standalone GPU correctness evidence only; original kernel is the complete bitwise regression reference, actual retained input buffers supply separate FP64 softmax samples.',
              cases=cases,exact_parent_word_pairs=exact,guard_word_observations=guards,
              inactive_zero_words=zero_values,independent_fp64_values=oracle_values,
              fixed_fp64_bound='1e-6 + 2e-6*abs(expected)',maximum_bound_fraction=max_fraction,
              rejected_control_variants=rejected,rejected_one_bit_corruptions=2,
              native_controls_pending_alias_and_lease_coverage='Not tested in a native session; required before integration.',
              speed_claim=False,quality_gate_passed=False,native_integration=False,goal_complete=False)
(s/'correctness-audit1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
