"""CPU admission tests; do not substitute for actual27B native qualification."""
import importlib.util
from pathlib import Path
import pytest

source=Path(__file__).resolve().parents[2]/'runtime/qwen38_serving.py'
spec=importlib.util.spec_from_file_location('qwen38_serving_contract',source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def profile():
    return {'batch':1,'maxInputTokens':3072,'maxOutputTokens':1024,'prefillChunk':32,
            'decodeChunk':4,'deviceId':0,'cudaGraph':True}

def test_supported_profile_has_explicit_bounded_context():
    module.validate_profile(profile())

@pytest.mark.parametrize('change',[{'maxInputTokens':28672,'maxOutputTokens':4096},
    {'batch':True},{'batch':4},{'deviceId':1},{'cudaGraph':False},{'imageGrid':[16,16]},
    {'speculativeMTP':3},{'maxOutputTokens':2048,'maxInputTokens':2048}])
def test_unqualified_profiles_cannot_silently_select_another_native_route(change):
    with pytest.raises(ValueError):module.validate_profile({**profile(),**change})
