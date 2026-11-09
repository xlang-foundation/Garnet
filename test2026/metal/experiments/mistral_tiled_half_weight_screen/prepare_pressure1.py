"""Add an outside-timed-operation cache-pressure control; preserve all math."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=(ROOT/'combined1.metal').read_text()
sweep='''
kernel void cache_pressure(device const uint4* input [[buffer(0)]],device uint4* output [[buffer(1)]],
 constant uint2& dims [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
 if(tid<dims.x){uint4 v=input[tid];output[tid]=((v<<1)|(v>>31))+uint4(tid^dims.y,tid+3u,tid+7u,tid+11u);}
}
'''
shader=ROOT/'pressure1.metal';assert not shader.exists();shader.write_text(original+sweep)
text=(ROOT/'screen1.mm').read_text();changes=[]
def edit(a,b):
    global text
    assert text.count(a)==1,(a[:90],text.count(a));text=text.replace(a,b);changes.append(dict(old=a,new=b))
edit('destination=argv[4];Need(!fs::exists(destination)', 'Need(timing,"Pressure host is timing-only");destination=argv[4];Need(!fs::exists(destination)')
edit(' auto gather=pipe(@"packed_affine_gather");',' auto pressurePipe=pipe(@"cache_pressure");\n auto gather=pipe(@"packed_affine_gather");')
edit('  auto encode=[&](size_t rows,','''  constexpr size_t sweepBytes=256*1024*1024;
  Need(device.recommendedMaxWorkingSetSize>sweepBytes*2+256*1024*1024&&device.currentAllocatedSize<=device.recommendedMaxWorkingSetSize-sweepBytes*2-256*1024*1024,"Cache pressure exceeds allocation envelope");
  auto sweepA=buffer(sweepBytes),sweepB=buffer(sweepBytes);uint32_t sweepSequence=0;double sweepStart=0,sweepEnd=0;
  {auto cb=[queue commandBuffer];auto e=[cb blitCommandEncoder];for(auto b:{sweepA,sweepB})[e fillBuffer:b range:NSMakeRange(0,sweepBytes) value:0];[e endEncoding];finish(cb);}
  auto encode=[&](size_t rows,''')
edit('   auto cb=[queue commandBuffer];if(zeroInactive', '''   {auto command=[queue commandBuffer];auto enc=[command computeCommandEncoder];uint32_t dims[]={uint32_t(sweepBytes/16),++sweepSequence};
    [enc setComputePipelineState:pressurePipe];[enc setBuffer:sweepA offset:0 atIndex:0];[enc setBuffer:sweepB offset:0 atIndex:1];[enc setBytes:dims length:sizeof(dims) atIndex:2];[enc dispatchThreads:MTLSizeMake(sweepBytes/16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];finish(command);sweepStart=command.GPUStartTime;sweepEnd=command.GPUEndTime;std::swap(sweepA,sweepB);}
   auto cb=[queue commandBuffer];if(zeroInactive''')
edit('{"gpu_end_seconds",cb.GPUEndTime}});','{"gpu_end_seconds",cb.GPUEndTime},{"sweep_gpu_start_seconds",sweepStart},{"sweep_gpu_end_seconds",sweepEnd},{"sweep_bytes_read",sweepBytes},{"sweep_bytes_written",sweepBytes}});')
host=ROOT/'pressure1.mm';assert not host.exists();host.write_text(text)
plan=dict(passed=True,reason='Initial warm operator observations were slower with prepared weights. The full native model cycles a much larger working set; inspect sensitivity to an explicit256MiB read/write pressure control before closing the cache hypothesis.',qualified_shader_prefix_sha256=sha(ROOT/'combined1.metal'),new_shader_sha256=sha(shader),original_host_sha256=sha(ROOT/'screen1.mm'),pressure_host_sha256=sha(host),host_changes=changes,sweep_bytes_read=256*1024*1024,sweep_bytes_written=256*1024*1024,pressure_is_outside_model_command_buffer=True,no_claim_all_hardware_caches_cold=True,native_runtime_changed=False,goal_complete=False)
(ROOT/'pressure-derivation1.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(prepared=True,shader_sha256=sha(shader),host_sha256=sha(host))))
