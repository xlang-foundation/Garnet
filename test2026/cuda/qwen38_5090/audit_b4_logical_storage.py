"""Audit logical B4 state/output storage from closed, hash-bound evidence (no GPU)."""
from pathlib import Path
import ast,hashlib,json,sys,zipfile
out=Path(sys.argv[1]);sha=lambda b:hashlib.sha256(b).hexdigest()
name='shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1'
proof=json.loads((out/(name+'-verified.json')).read_bytes());raw=(out/(name+'-evidence.zip')).read_bytes()
assert proof['model_gate_passed'] and proof['archive_sha256']==sha(raw)
with zipfile.ZipFile(out/(name+'-evidence.zip')) as z:
 manifest=json.loads(z.read('evidence-manifest.json'))
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 source=z.read('Garnet/xModel/qwen35/vl_0_8b/cuda_packed_program.py');tree=ast.parse(source)
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decoder_state_layout')
 ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),'archived-layout','exec'),ns)
 report=json.loads(z.read('results/'+name+'/native/report.json'))
parent='batch-mtp-host-embedding-http-pilot-attempt1';parent_proof=json.loads((out/(parent+'-verified.json')).read_bytes())
assert sha((out/(parent+'-evidence.zip')).read_bytes())==parent_proof['archive_sha256']
with zipfile.ZipFile(out/(parent+'-evidence.zip')) as z:
 config_raw=z.read('models/Qwen3.8-27B-NVFP4/config.json');config=json.loads(config_raw)['text_config']
layout,width=ns['decoder_state_layout'](config,4,4096)
by_kind={kind:sum(layer['state1_width']+layer['state2_width'] for layer in layout if layer['kind']==kind)*4*4 for kind in ('linear_attention','full_attention')}
assert len(layout)==64 and sum(by_kind.values())==width*4*4==2782920704
cases=[]
for c in report['cases']:
 memory=c['decoder_memory'];assert memory['state_bytes']==sum(by_kind.values())
 cases.append({'group':c['group'],'repeat':c['repeat'],'canonical_state_bytes':memory['state_bytes'],'graph_owned_output_bytes':sum(g['owned_device_output_bytes'] for g in memory['graphs']),'compressed_pool_bytes':memory['compressed_projection_weight_bytes']+memory['compressed_projection_scale_bytes'],'host_embedding_mapped_bytes':memory['host_embedding_mapped_tensor_bytes']})
result={'status':'closed_b4_logical_storage_audit','full_model_archive_sha256':proof['archive_sha256'],'configuration_parent_archive_sha256':parent_proof['archive_sha256'],'config_sha256':sha(config_raw),'layout_source_sha256':sha(source),'batch':4,'capacity':4096,'state_bytes_by_kind':by_kind,'cases':cases,'scope':'Logical retained storage only, not measured DRAM traffic or complete peak decomposition. Graph output contains full state plus hidden/payload. TensorRT runtime tracking was disabled. Full KV output/packing/canonical copies are a profiling hypothesis; compiler fusion and actual traffic require measurement. No speed/causal claim.','next_diagnostic':'After current HTTP pilot closes, profile target execution and state commit separately. Any compact/in-place KV path must preserve accepted-prefix/rollback, inactive rows, reset, ordinary/MTP state and IDs before timing.'}
(out/'Garnet-B4-Logical-Storage-Audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

