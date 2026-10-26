"""Cold embedded-process constructor observation; no Garnet import or model inference."""
from pathlib import Path
import json,os,sys
root,out=map(Path,sys.argv[1:3]);assert not out.exists();out.mkdir();sys.path.insert(0,str(root/'Garnet'))
from test2026.cuda.qwen38_5090.fp8_z128_audit_bridge import observe
preloaded={name:name in sys.modules for name in ('garnet','tensor')}
result=observe(root,out/'context-audit.json')
(out/'report.json').write_bytes(json.dumps({'status':'cold_actual_helper_audit_observed','observation':result,'host_preloaded_modules':preloaded,'probe_explicit_Garnet_import':False,'model_or_performance_qualified':False},indent=2).encode());print(json.dumps({'status':'cold_actual_helper_audit_observed','validation':result['validation']}))
