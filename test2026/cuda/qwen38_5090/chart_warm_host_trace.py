"""Plot disjoint top-level host intervals from an independently replayed trace."""
from pathlib import Path
import hashlib,json,sys,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
source,destination=map(Path,sys.argv[1:3]);name='kv-delta-warm-host-trace-attempt1'
sha=lambda b:hashlib.sha256(b).hexdigest();proof_path=source/(name+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='compact_warm_host_trace_replayed' and proof['raw_hidden_all_exact'] and proof['host_intervals']==1148
archive=source/(name+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
with zipfile.ZipFile(archive) as z:
    records=[json.loads(line) for line in z.read('results/'+name+'/native/stages.jsonl').splitlines()]
    report=json.loads(z.read('results/'+name+'/native/correctness/report.json'))
    resource=json.loads(z.read('results/'+name+'/resources/resource-monitor.json'))
stages=('target_verify','draft','prefix_commit','draft_correct','other')
labels=('Target verification*','Draft proposal','Prefix commit','Draft correction','Other generate work')
colors=('#1968b4','#df9939','#7551a8','#3b9576','#aab4c2');data=[]
for g in (0,1):
    for r in (2,3):
        tag=f'group-{g}-repeat-{r}';selected=[row for row in records if row['tag']==tag]
        outer=[row for row in selected if row['stage']=='generate' and row['parent'] is None];assert len(outer)==1
        outer=outer[0];children=[row for row in selected if row['parent']==outer['number']]
        assert set(row['stage'] for row in children)==set(stages[:-1])
        intervals={stage:sum(row['host_wall_seconds'] for row in children if row['stage']==stage) for stage in stages[:-1]}
        total=outer['host_wall_seconds'];intervals['other']=total-sum(intervals.values());assert intervals['other']>=0
        case=next(c for c in report['cases'] if c['group']==g and c['repeat']==r)
        data.append({'tag':tag,'group':g,'repeat':r,'total_generate_host_seconds':total,'disjoint_top_level_host_seconds':intervals,
                     'keys':case['keys'],'output_tokens_per_real_row':[len(v) for v in case['generated_ids']],
                     'target_verification_host_fraction':intervals['target_verify']/total})
destination.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,6));fig.set_facecolor('#f7f9fc')
for ax,g,unit,scale,title in zip(axes,(0,1),('Seconds','Milliseconds'),(1,1000),('Mixed text/image · 128 output tokens per row','VLM color answers · short outputs')):
    rows=[row for row in data if row['group']==g]
    left=[0.,0.]
    for stage,label,color in zip(stages,labels,colors):
        values=[row['disjoint_top_level_host_seconds'][stage]*scale for row in rows]
        ax.barh([0,1],values,left=left,color=color,label=label,height=.46);left=[a+b for a,b in zip(left,values)]
    for y,(value,row) in enumerate(zip(left,rows)):
        ax.text(value*1.015,y,f'{value:.3f}',va='center',fontsize=10,fontweight='bold')
        ax.text(value*.35,y,f"{row['target_verification_host_fraction']:.1%}",color='white',ha='center',va='center',fontweight='bold')
    ax.set_yticks([0,1],['Capture repeat 2','Capture repeat 3']);ax.invert_yaxis();ax.set_xlim(0,max(left)*1.2)
    ax.set_xlabel(unit+' of instrumented host interval');ax.set_title(title,loc='left',fontweight='bold',fontsize=11)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
handles,texts=axes[0].get_legend_handles_labels();fig.legend(handles,texts,loc='upper center',bbox_to_anchor=(.52,.845),ncol=5,frameon=False,fontsize=9)
fig.suptitle('Warm execution diagnostics · host API intervals',x=.055,y=.98,ha='left',fontsize=19,fontweight='bold')
fig.text(.055,.915,'One RTX 5090 · physical B4/T4 · MTP3 · 8 batched correctness cases / 16 raw hidden arrays replayed · 4 captured generate calls',fontsize=10,color='#45536a')
fig.text(.055,.16,'Only disjoint top-level intervals are stacked. Nested program timings remain in the raw trace. Two warmups precede each group.',fontsize=9,color='#45536a')
fig.text(.055,.105,'*Verification includes completion of pending GPU work and head readback. Host interval attribution does not isolate GPU operators.',fontsize=9,color='#45536a')
fig.text(.055,.05,'No added per-stage GPU barriers or log writes; tracing still has overhead. Not CUDA kernel time, HTTP latency, a speedup or a 2× result.',fontsize=9,color='#45536a')
fig.subplots_adjust(left=.12,right=.97,top=.73,bottom=.29,wspace=.4)
artifacts={}
for ext in ('png','svg'):
    p=destination/('Garnet-Warm-Host-Stage-Trace.'+ext);fig.savefig(p,dpi=170,facecolor=fig.get_facecolor());artifacts[p.name]=sha(p.read_bytes())
plt.close(fig)
provenance={'status':'verified_warm_host_diagnostic_chart_generated','archive_sha256':proof['archive_sha256'],'proof_sha256':sha(proof_path.read_bytes()),
    'generator_sha256':sha(Path(__file__).read_bytes()),'all_capture_tags':[row['tag'] for row in data],'data':data,
    'sampled_correctness_peak_adapter_used_bytes':resource['sampled_memory']['sampled_peak_adapter_used_bytes'],'artifacts':artifacts,
    'performance_admitted':False,'scope':proof['scope']+' Plot stacks disjoint top-level host intervals only; nested child intervals are not added twice.'}
(destination/'Garnet-Warm-Host-Stage-Trace-Provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
print(json.dumps({'captures':len(data),'artifacts':list(artifacts),'verification_fractions':[row['target_verification_host_fraction'] for row in data]}))
