"""CPU-only real embedded bridge protocol exercised under Nsight Systems."""
import ctypes
import json
import sys
from pathlib import Path

bridge = ctypes.CDLL('D:/CantorAI/work/qwen35-cuda/qwen35_nvtx_bridge_v1.dll')
bridge.qwen35NvtxPush.argtypes = [ctypes.c_char_p]
bridge.qwen35NvtxPush.restype = ctypes.c_int
assert bridge.qwen35NvtxVersion() == 3
thread = bridge.qwen35NvtxThreadId()
events = []
for label in ('bridge_selftest_outer', 'bridge_selftest_inner'):
    events.append({'operation': 'push', 'label': label, 'result': bridge.qwen35NvtxPush(label.encode('ascii')),
                   'thread': bridge.qwen35NvtxThreadId()})
for label in ('bridge_selftest_inner', 'bridge_selftest_outer'):
    events.append({'operation': 'pop', 'label': label, 'result': bridge.qwen35NvtxPop(),
                   'thread': bridge.qwen35NvtxThreadId()})
assert all(e['thread'] == thread for e in events)
output = Path(sys.argv[1])
assert not output.exists()
output.write_text(json.dumps({'version': 3, 'thread_id': thread, 'events': events,
    'scope': 'CPU-only native bridge calls; actual NVTX SQLite event names/order/thread still require audit.'}, indent=2)+'\n')
print('QWEN35_NVTX_CPU_BRIDGE_CALLS_COMPLETED', thread, flush=True)
