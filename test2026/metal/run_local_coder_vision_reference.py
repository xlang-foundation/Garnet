"""Native learned vision parity. XLang3 args: fixture folder, build, source, report.

This standalone gate must run without the text decoder or any comparator.
"""
import sys,json,math,time
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
with open(sys.argv[1]+'/fixture.json') as stream:fixture=json.load(stream)
io=garnet.tensor_io();vision=None
report=dict(status='running',scope=fixture['scope'],backend=garnet.backend_info())
try:
    io.open(fixture['checkpoint'])
    assert io.checkpoint_format=='mlx_qwen35_affine_int4_g64_bf16'
    options=dict(patch_size=16,merge_size=2,temporal_patch_size=2,pixels=65536,
                 image_mean=fixture['processor']['image_mean'],image_std=fixture['processor']['image_std'])
    prepared=garnet.prepare_image(fixture['image'],json.dumps(options))
    assert [prepared[1],prepared[2]]==fixture['grid']
    expected_patches=io.read_f32(sys.argv[1]+'/patches.f32',fixture['patches_shape']).tolist()
    patches=prepared[0].tolist()
    assert len(patches)==len(expected_patches)
    patch_error=0
    for x,y in zip(patches,expected_patches):
        assert math.isfinite(x) and math.isfinite(y)
        patch_error=max(patch_error,abs(x-y))
    assert patch_error<=1e-6,patch_error
    report['patch_max_abs']=patch_error
    report['memory_before']=json.loads(garnet.memory_stats_json())
    began=time.monotonic()
    vision=MetalImageEncoder(fixture['checkpoint'],fixture['config'],16,16,1)
    report['build_seconds']=time.monotonic()-began
    report['memory_after_build']=json.loads(garnet.memory_stats_json())
    began=time.monotonic();actual=vision.encode_patches(prepared[0]).tolist()
    report['encode_seconds']=time.monotonic()-began
    expected=io.read_f32(sys.argv[1]+'/expected.f32',fixture['expected_shape']).tolist()
    assert len(actual)==len(expected)==64*5120
    maximum=0;squared=0;dot=0;aa=0;bb=0;failures=0
    for x,y in zip(actual,expected):
        assert math.isfinite(x) and math.isfinite(y)
        error=abs(x-y);maximum=max(maximum,error);squared+=error*error
        dot+=x*y;aa+=x*x;bb+=y*y
        if error>0.003+0.001*abs(y):failures+=1
    report.update(compared_values=len(actual),max_abs=maximum,rmse=math.sqrt(squared/len(actual)),
                  cosine=dot/math.sqrt(aa*bb),elementwise_failures=failures,
                  tolerance=dict(atol=0.003,rtol=0.001,min_cosine=0.99999))
    assert failures==0 and report['cosine']>=0.99999,report
    report['status']='passed_learned_vision_reference_not_end_to_end'
    print('native_learned_vision_reference PASS',maximum,report['cosine'],flush=True)
except:
    report['status']='failed'
    raise
finally:
    if vision is not None:vision.close()
    io.close()
    report['memory_after_close']=json.loads(garnet.memory_stats_json())
    with open(sys.argv[4],'w') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    garnet.shutdown()
