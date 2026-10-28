"""Scientific diagnostic chart from replayed markers and audited operation counts."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root,destination=map(Path,sys.argv[1:3]);sha=lambda b:hashlib.sha256(b).hexdigest()
proof_path=root/'kv-delta-decode-cuda-trace-attempt1-verified.json';proof=json.loads(proof_path.read_bytes())
graph_path=root/'Garnet-Decode-CUDA-Graph-Analysis.json';graph=json.loads(graph_path.read_bytes())
audit_path=root/'Garnet-Decode-CUDA-Operation-Counts.json';audit=json.loads(audit_path.read_bytes())
assert proof['status']=='compact_decode_cuda_trace_replayed' and proof['raw_hidden_all_exact']
assert graph['status']=='marked_decode_cuda_graphs_reconstructed' and audit['status']=='selected_decode_operation_counts_matched'
assert graph['proof_sha256']==audit['proof_sha256']==sha(proof_path.read_bytes())
assert graph['archive_sha256']==audit['archive_sha256']==proof['archive_sha256']
families=[('NVFP4 matrix kernels',lambda n:n.startswith('cutlass3x_sm120_bstensorop_')),
 ('FP8 matrix + split-K kernels',lambda n:n.startswith('nvjet_sm120_') or 'sm89_xmma_gemm_e4m3' in n),
 ('BF16 matrix kernels',lambda n:'cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_' in n),
 ('GDN recurrence',lambda n:'::recurrence(' in n),
 ('Short-context attention',lambda n:'::compactCached<(bool)0>(' in n),
 ('Canonical state commit',lambda n:('::copyLinear(' in n or '::scatterKV(' in n) and 'Qwen35KVDelta' in n),
 ('NVFP4 / FP8 quantization',lambda n:'::QuantizeActivation(' in n or '::Quantize(' in n)]
rows=[]
for w in proof['windows']:
    counts=[0]*len(families);other=0
    for kernel in w['kernels_by_name']:
        matches=[i for i,(_,match) in enumerate(families) if match(kernel['name'])];assert len(matches)<=1
        if matches:counts[matches[0]]+=kernel['sum_duration_ns']
        else:other+=kernel['sum_duration_ns']
    assert sum(counts)+other==w['kernel_sum_duration_ns']
    rows.append(counts+[other])
values=np.array(rows)/1e6;labels=[f[0] for f in families]+['Other recorded kernels']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,(left,right)=plt.subplots(1,2,figsize=(15.5,7),gridspec_kw={'width_ratios':[1,1.45]});fig.set_facecolor('#f7f9fc')
colors=['#2166ac','#67a9cf','#d7dfe9'];parts=[]
for w in graph['windows']:
    assert w['kernel_union_ns']<=w['kernel_copy_memset_union_ns']<=w['marker_span_ns']
    parts.append([w['kernel_union_ns'],w['kernel_copy_memset_union_ns']-w['kernel_union_ns'],w['unobserved_interval_ns']])
parts=np.array(parts)/1e6;bottom=np.zeros(2)
for i,title in enumerate(('Kernel interval union','Copy/memset outside kernels','Remaining interval')):
    left.bar([0,1],parts[:,i],bottom=bottom,color=colors[i],width=.55,label=title);bottom+=parts[:,i]
for i,total in enumerate(bottom):left.text(i,total+35,f'{total:.1f} ms',ha='center',fontweight='bold',color='#18304b')
left.set_xticks([0,1],['Warm repeat 2','Warm repeat 3']);left.set_ylabel('Marker interval · milliseconds');left.set_title('Observed decode interval',loc='left',fontweight='bold');left.set_ylim(0,max(bottom)*1.15);left.grid(axis='y',alpha=.15);left.set_axisbelow(True)
left.legend(loc='upper left',bbox_to_anchor=(0,-.13),frameon=False,fontsize=9)
y=np.arange(len(labels));height=.34
for i,color in enumerate(('#2166ac','#76b2cf')):
    right.barh(y+(i-.5)*height,values[i],height=height,color=color,label='Warm repeat '+str(i+2))
    for at,value in zip(y+(i-.5)*height,values[i]):right.text(value+5,at,f'{value:.1f}',va='center',fontsize=8,color='#18304b')
right.set_yticks(y,labels);right.invert_yaxis();right.set_xlim(0,max(values.max()*1.16,1));right.set_xlabel('Summed kernel duration · milliseconds');right.set_title('Recorded kernel families',loc='left',fontweight='bold');right.grid(axis='x',alpha=.15);right.set_axisbelow(True);right.legend(loc='upper right',bbox_to_anchor=(1,1.13),ncol=2,frameon=False,fontsize=9)
fig.suptitle('Warm decode CUDA profile · single RTX 5090',x=.055,y=.97,ha='left',fontsize=20,fontweight='bold')
fig.text(.055,.9,'Qwen3.8-27B mixed NVFP4/FP8 · physical B4 · mixed text/image · MTP3 / T4 · two generate calls after prefill',fontsize=10,color='#45536a')
fig.text(.055,.135,'All 8 correctness cases and 16 raw hidden arrays match retained controls; unique HtoD markers and stable graph captures verified.',fontsize=9,color='#45536a')
fig.text(.055,.09,'Profiler overhead applies. Two auxiliary CPU file reads recorded. API waits overlap GPU work; durations must not be added.',fontsize=9,color='#45536a')
fig.text(.055,.045,'Kernel-family sums may overlap; remaining intervals do not prove device idleness. Diagnostic only: no HTTP speedup or established 2× claim.',fontsize=9,color='#45536a')
fig.subplots_adjust(left=.07,right=.97,top=.8,bottom=.33,wspace=.75)
destination.mkdir(parents=True,exist_ok=True);artifacts={}
for ext in ('png','svg'):
    p=destination/('Garnet-Warm-Decode-CUDA-Profile.'+ext);fig.savefig(p,dpi=170,facecolor=fig.get_facecolor());artifacts[p.name]=sha(p.read_bytes())
plt.close(fig)
provenance={'status':'verified_decode_cuda_chart_generated','generator_sha256':sha(Path(__file__).read_bytes()),
    'proof_sha256':sha(proof_path.read_bytes()),'graph_analysis_sha256':sha(graph_path.read_bytes()),'count_audit_sha256':sha(audit_path.read_bytes()),
    'archive_sha256':proof['archive_sha256'],'family_labels':labels,'family_sum_ms':values.tolist(),'marker_union_parts_ms':parts.tolist(),
    'performance_admitted':False,'two_x_established':False,'artifacts':artifacts}
(destination/'Garnet-Warm-Decode-CUDA-Profile-Provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
print(json.dumps({'charts':1,'artifacts':artifacts}))
