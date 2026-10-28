# SPDX-License-Identifier: Apache-2.0
"""Static audit only; does not build or execute Metal or infer numeric parity."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo=Path(__file__).resolve().parents[2];output=Path(sys.argv[1])
refs=('5b0dc7ff8^','5b0dc7ff8')
paths=('xModel/qwen35/vl_4b/model.py','xModel/qwen35/vl_4b/serving-profile.json',
       'xModel/qwen35/vl_0_8b/metal_decoder.py','xModel/qwen35/vl_0_8b/metal_vision.py',
       'runtime/metal_serving.py','cmake/GarnetMetal.cmake')
def sha(data):return hashlib.sha256(data).hexdigest()
def calls(data):
    module=ast.parse(data.decode('utf-8-sig'));result={}
    for node in module.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('create_decoder','create_vision'):
            constructors=[n.value for n in ast.walk(node) if isinstance(n,ast.Return) and isinstance(n.value,ast.Call)]
            assert len(constructors)==1
            result[node.name]={'arguments':ast.dump(node.args,include_attributes=False),
                               'returned_constructor':ast.dump(constructors[0],include_attributes=False)}
    return result
working={p:(repo/p).read_bytes() for p in paths};baseline=[]
for ref in refs:
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse',ref],text=True).strip();files=[];factory=None
    for name in paths:
        old=subprocess.run(['git','-C',str(repo),'show',revision+':'+name],capture_output=True)
        if old.returncode:
            files.append({'path':name,'present_in_reference':False,'working_sha256':sha(working[name])});continue
        files.append({'path':name,'present_in_reference':True,'reference_sha256':sha(old.stdout),
                      'working_sha256':sha(working[name]),'whole_file_equal':old.stdout==working[name]})
        if name.endswith('/model.py'):factory={'reference':calls(old.stdout),'working':calls(working[name])}
    assert factory is not None
    baseline.append({'requested_ref':ref,'revision':revision,'files':files,'metal_constructor_calls_and_signatures_equal':factory['reference']==factory['working'],'factory_comparison':factory})
cmake=(repo/'CMakeLists.txt').read_text(encoding='utf-8')
isolated=('include(cmake/GarnetMetal.cmake)\n    return()' in cmake and
          cmake.index('include(cmake/GarnetMetal.cmake)')<cmake.index('project("garnet" LANGUAGES CXX CUDA)'))
result={'scope':'Static source/constructor audit against the pushed WorldSense checkpoint and its parent. No remote Mac action or Metal build/inference.',
        'references':baseline,'metal_cmake_returns_before_cuda_project':isolated,
        'fresh_metal_execution_verified':False,
        'limitations':'Equal constructor arguments or files do not prove complete current-source Metal compilation/numerical behavior. Changed files, if any, remain visible for review.'}
output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'factory_equal_by_ref':{r['requested_ref']:r['metal_constructor_calls_and_signatures_equal'] for r in baseline},'metal_cmake_isolated':isolated,'metal_executed':False}))
