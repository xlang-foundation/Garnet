"""Show audited stage kernel sums; explicitly not end-to-end benchmark times."""
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root=Path(sys.argv[1]).resolve();source=root/'qwen35-nvtx-stage-v1-summary.json'
raw=source.read_bytes();data=json.loads(raw)
assert data['all56_native_work_owner_and_stage_controls_passed'] and data['process_observation_qualified']
assert not data['primary_latency_accepted'] and not data['memory_accepted']
stem=root/'progress-warmed-nvtx-stages-v1'
assert not any(Path(str(stem)+s).exists() for s in ('.png','.svg','-sources.json'))
groups=[('Target verification','#236eb4'),('Target prefill','#48a3c7'),('MTP','#eda33d'),
        ('Prefix commit','#55a67b'),('Image / input','#8c79b7'),('Other','#c8d0db')]
cases=[]
for case in data['cases']:
    stages=case['stage_trace']['stages'];sums=[0.0]*6
    for name,stage in stages.items():
        if name=='target_verification':index=0
        elif name=='target_prefill':index=1
        elif name.startswith('mtp_'):index=2
        elif name=='target_prefix_commit':index=3
        elif name=='input_preparation':index=4
        else:index=5
        sums[index]+=stage['kernel']['sum_ms']
    assert abs(sum(sums)-case['stage_trace']['totals']['kernel']['sum_ns']/1e6)<1e-7
    cases.append({'model':case['model'],'batch':case['batch'],'kernel_sum_ms':sum(sums),
                  'groups_ms':dict(zip((g[0] for g in groups),sums))})
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
fig=plt.figure(figsize=(12,5.1),facecolor='white');ax=fig.add_axes([.14,.27,.79,.43])
for y,case in zip((1,0),cases):
    left=0
    for (label,color),value in zip(groups,case['groups_ms'].values()):
        ax.barh(y,value,left=left,height=.36,color=color,edgecolor='white',linewidth=.5)
        if value>=45:ax.text(left+value/2,y,f'{value:.1f}',ha='center',va='center',color='white',weight='bold')
        left+=value
    ax.text(left+7,y,f'{left:.1f} ms',va='center',weight='bold',color='#243449')
ax.set_yticks([1,0],['2B / batch 1','4B / batch 8']);ax.set_ylim(-.65,1.65);ax.set_xlim(0,620)
ax.set_xlabel('Sum of GPU kernel durations (milliseconds)');ax.set_axisbelow(True);ax.grid(axis='x',color='#e6ebf0')
ax.tick_params(axis='both',length=0,pad=8)
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.055,.94,'Target decoder work dominates the warmed GPU trace',fontsize=18,weight='bold',color='#172b43')
fig.text(.055,.88,'Original MTP policy · one warmed request per model · all 56 request controls passed',color='#526174')
fig.legend(handles=[Patch(color=color,label=label) for label,color in groups],loc='upper left',
           bbox_to_anchor=(.14,.815),ncol=3,frameon=False,fontsize=10)
fig.text(.055,.14,'100% of kernel count and summed time attributed through CUDA API correlation and actual NVTX nesting.',fontsize=10,color='#526174')
fig.text(.055,.095,'Original owned allocations and one MTP context preserved. Copies and API waits are reported separately.',fontsize=10,color='#526174')
fig.text(.055,.05,'Kernel sums can overlap; they are not request latency or removable overhead. Different batches; no primary speed or 2x claim.',fontsize=9,color='#526174')
fig.savefig(str(stem)+'.png',dpi=160);fig.savefig(str(stem)+'.svg');plt.close(fig)
Path(str(stem)+'-sources.json').write_text(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),
    'source_file':str(source),'execution_identity_sha256':data['execution_identity_sha256'],'cases':cases,
    'primary_latency_accepted':False,'scope':'Single-request instrumented GPU kernel sums, not benchmark latency or speedup.'},indent=2)+'\n')
print(str(stem)+'.png')
