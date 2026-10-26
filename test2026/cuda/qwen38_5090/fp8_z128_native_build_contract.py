"""Pure CPU validation of the actual separate Z plugin contract report."""
from pathlib import PureWindowsPath


def z_wire_contract(report):
    values={'status':'fp8_z128_plugin_cpu_contract_passed','wire_bytes':20,'wire_policy':0x5a313238,'bad_wires':34,'bad_constructors':18,'bad_descriptors':23,'bad_formats':11,'policy_rejections':8,'cross_wire_rejections':5,'workspace_bytes':4849664,'unexpected_gpu_calls':0,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False}
    for key,value in values.items():assert type(report[key]) is type(value) and report[key]==value,key
    assert type(report['runtime_version']) is type(report['header_version']) is int
    assert report['runtime_version']==report['header_version']==101601
    assert PureWindowsPath(report['loaded_tensorrt_dll'])==PureWindowsPath('D:/CantorAI/5090-study-20261023/tools/TensorRT/bin/nvinfer_10.dll')
    return values
