"""Check actual NVTX event identity/nesting, not return-depth conventions."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = root / 'qwen35-nvtx-cpu-selftest-v1-summary.json'
assert not out.exists()
stem = root / 'qwen35-nvtx-cpu-selftest-v1'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
assert read(Path(str(stem)+'-exit.json'))['exit_code'] == 0
assert read(root/'qwen35-nvtx-bridge-v1-exit.json')['exit_code'] == 0
raw = read(Path(str(stem)+'.json'))
assert raw['version'] == 3
assert [e['operation'] for e in raw['events']] == ['push', 'push', 'pop', 'pop']
assert all(e['thread'] == raw['thread_id'] for e in raw['events'])
c = sqlite3.connect(Path(str(stem)+'.sqlite').as_uri()+'?mode=ro', uri=True)
events = list(c.execute('select start,end,eventType,text,globalTid from NVTX_EVENTS order by start'))
assert len(events) == 2
outer, inner = events
assert outer[2:4] == (59, 'bridge_selftest_outer') and inner[2:4] == (59, 'bridge_selftest_inner')
assert outer[0] < inner[0] < inner[1] < outer[1] and outer[4] == inner[4]
paths = [Path(str(stem)+s) for s in ('.json', '.sqlite', '.nsys-rep', '-exit.json')]
paths += [root/'qwen35_nvtx_bridge_v1.dll', root/'qwen35-nvtx-bridge-v1-build-inputs.json', root/'qwen35-nvtx-bridge-v1-exit.json']
out.write_text(json.dumps({'actual_nvtx_names_nesting_and_single_thread_passed': True,
    'events': events, 'native_thread_id': raw['thread_id'],
    'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'scope': 'CPU-only bridge/actual Nsight NVTX protocol. Return values -1 are retained; actual captured ranges prove annotation, not GPU/model behavior.'}, indent=2)+'\n')
print('QWEN35_NVTX_CPU_REAL_EVENT_AUDIT_PASS')
