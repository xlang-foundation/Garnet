"""Emit independent layout tables for a standalone native operator test.

This imports no Garnet/CUDA modules and never opens a GPU. The executable must
be run separately on the dedicated RTX5090 after all live workers are closed.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[3]
source=repo/'xModel/qwen35/vl_0_8b/kv_bf16_study_layout.py'
spec=importlib.util.spec_from_file_location('bf16_layout_fixture',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
destination=Path(sys.argv[1]);assert not destination.exists()
text=['// Generated from the separate Python byte-layout contract.\n#pragma once\n',
      'struct BF16LayerFixture { long long canonical,target,replay,width; bool full; };\n']
layouts={}
for chunk in (4,32):
    layout=module.build_bf16_kv_study_layout(chunk=chunk,retain_linear_inputs=chunk==4)
    layouts[str(chunk)]=layout
    text.append('static const BF16LayerFixture bf16Layers'+str(chunk)+'[64]={\n')
    for layer in layout['layers']:
        values=(layer['offset'],layer['target_offset'],layer['replay_offset'] or 0,
                layer['state1_width']+layer['state2_width'])
        text.append(' {'+','.join(map(str,values))+','+str(layer['kind']=='full_attention').lower()+'},\n')
    text.append('};\n')
raw=''.join(text).encode();destination.write_bytes(raw)
print(json.dumps({'status':'bf16_native_fixture_tables_prepared_unrun',
    'layout_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'header_sha256':hashlib.sha256(raw).hexdigest(),
    'header_path':str(destination),'runtime_qualified':False},indent=2))
