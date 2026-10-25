"""Publication-style pilot charts from hash-bound, independently replayed data.

Usage: python chart_kv_delta_http.py EVIDENCE_DIRECTORY OUTPUT_DIRECTORY
Never substitutes example numbers or omits unfavorable cohorts.
"""
from pathlib import Path
import hashlib,json,math,sys,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

source,destination=map(Path,sys.argv[1:3]);sha=lambda b:hashlib.sha256(b).hexdigest()
pilot='register128-padded-http-pilot-attempt1';parent='padded-mtp-http-pilot-attempt1'
proof_path=source/(pilot+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='register128_padded_http_pilot_replayed' and proof['measured_sse_worker_pairs']==64 and not proof['two_x_established']
assert proof['grouping_clock']=='perf_counter'
parent_path=source/(parent+'-verified.json');old=json.loads(parent_path.read_bytes());assert old['status']=='padded_mtp_http_pilot_replayed'
archives={pilot+'-evidence.zip':proof['archive_sha256'],parent+'-evidence.zip':old['archive_sha256']}
with zipfile.ZipFile(source/(pilot+'-evidence.zip')) as z:
    manifest=json.loads(z.read('evidence-manifest.json'));archives.update(manifest['linked_archives'])
    assert old['archive_sha256']==manifest['linked_archives'][parent+'-evidence.zip']
for name,digest in archives.items():assert sha((source/name).read_bytes())==digest
assert all(c['identical_request_bytes'] and c['identical_transport'] and c['identical_prompt_output_counts'] for c in proof['comparison']['cohorts'])
cases=('text','image');concurrency=(1,2,4,8);cohorts=[case+'-c'+str(c) for case in cases for c in concurrency]
series={
    'compact':{'label':'Garnet register128 B4','color':'#1968b4','style':'-','run':proof['run'],'archive':pilot+'-evidence.zip'},
    'previous':{'label':'Garnet register-off B4','color':'#7d8796','style':'--','run':old['run'],'archive':parent+'-evidence.zip'},
}
for reference in proof['references']:
    engine=reference['engine'];assert engine in ('vllm','sglang')
    series[engine]={'label':'vLLM ordinary' if engine=='vllm' else 'SGLang ordinary',
        'color':'#d68b25' if engine=='vllm' else '#3b9576','style':'-','run':reference,
        'archive':'vllm-graph-linkfix-evidence.zip' if engine=='vllm' else 'sglang-corrected-ipv4-evidence.zip'}
assert set(series)=={'compact','previous','vllm','sglang'}
data={};gpu=None
for name,item in series.items():
    with zipfile.ZipFile(source/item['archive']) as z:
        paths=[n for n in z.namelist() if n.endswith('-manifest.json') and n.rsplit('/',1)[-1] not in ('evidence-manifest.json','preparation-manifest.json','source-manifest.json')]
        assert len(paths)==1;raw=json.loads(z.read(paths[0]));assert raw['context_capacity']==4096
        assert gpu is None or gpu==raw['gpu'];gpu=raw['gpu']
        peaks={c['case']+'-c'+str(c['concurrency']):c['sampled_peak_adapter_used_bytes']/2**30 for c in raw['cohorts']}
        assert set(item['run']['cohorts'])==set(peaks)==set(cohorts)
        data[name]={}
        for key in cohorts:
            row=item['run']['cohorts'][key];assert len(row['requests'])==8
            expected=41 if key.startswith('text') else 104
            assert all(v['prompt_tokens']==expected and v['generated_tokens']==128 for v in row['requests'].values())
            metrics=dict(row['metrics']);metrics['sampled_peak_adapter_gib']=peaks[key]
            assert all(isinstance(metrics[k],(int,float)) and math.isfinite(metrics[k]) and metrics[k]>0 for k in ('ttft_median_ms','decode_median_tokens_per_second','output_tokens_per_second','sampled_peak_adapter_gib'))
            data[name][key]=metrics
destination.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
artifacts=[]
common='Single RTX 5090 32GB · same checkpoint and request bytes · 41 text / 104 image prompt tokens · 128 outputs · 8 requests/cohort'
footer='Single pilots; untuned refs. Physical B4/MTP3 with private padding; C8 queues B4 groups. Both use QPC 10ms grouping; only recurrence selection changed.'
limitations='4096 capacity; actual 4K prompts, broad quality, tuned references and clean repeats remain pending. No established 2× claim.'
def save(fig,stem):
    for ext in ('png','svg'):
        p=destination/(stem+'.'+ext);fig.savefig(p,dpi=170,facecolor=fig.get_facecolor());artifacts.append(p)
    plt.close(fig)
definitions=(
    ('ttft_median_ms','Time to first token','Milliseconds · lower is better','Garnet-Register128-MTP-HTTP-TTFT'),
    ('decode_median_tokens_per_second','Request decode speed','Tokens / second · higher is better','Garnet-Register128-MTP-HTTP-Decode'),
    ('output_tokens_per_second','Concurrent output throughput','Total tokens / second · higher is better','Garnet-Register128-MTP-HTTP-Throughput'),
    ('sampled_peak_adapter_gib','Sampled GPU memory','GiB · adapter-wide sampled maximum','Garnet-Register128-MTP-HTTP-Memory'),
)
x=np.arange(4)
for metric,title,unit,stem in definitions:
    fig,axes=plt.subplots(1,2,figsize=(13,5.7));fig.set_facecolor('#f7f9fc')
    for ax,case in zip(axes,cases):
        maximum=0;positive=[]
        for name,item in series.items():
            values=[data[name][case+'-c'+str(c)][metric] for c in concurrency];positive+=values;maximum=max(maximum,max(values))
            ax.plot(x,values,color=item['color'],linestyle=item['style'],marker='o',markersize=4,linewidth=2 if name=='compact' else 1.5,label=item['label'])
            if name=='compact':
                for at,value in zip(x,values):ax.annotate(f'{value:.1f}',(at,value),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=item['color'],fontweight='bold')
        if metric!='sampled_peak_adapter_gib':
            target=[]
            for c in concurrency:
                values=[data[n][case+'-c'+str(c)][metric] for n in ('vllm','sglang')]
                target.append(min(values)/2 if metric=='ttft_median_ms' else max(values)*2)
            positive+=target;maximum=max(maximum,max(target));ax.plot(x,target,color='#a64048',linestyle=':',linewidth=1.5,label='Half fastest-reference TTFT' if metric=='ttft_median_ms' else '2× best reference (target)')
        ax.set_xticks(x,['C'+str(c) for c in concurrency]);ax.set_xlabel('Concurrent client requests');ax.set_title(case.title(),loc='left',fontweight='bold');ax.set_ylabel(unit)
        ax.set_xlim(-.12,3.12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        if metric=='ttft_median_ms':ax.set_yscale('log');ax.set_ylim(min(positive)*.7,maximum*1.5)
        else:ax.set_ylim(0,maximum*1.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.865),ncol=len(labels),frameon=False,fontsize=9)
    fig.suptitle(title+' · measured HTTP pilots',x=.055,y=.98,ha='left',fontsize=19,fontweight='bold');fig.text(.055,.915,common,fontsize=10,color='#45536a')
    fig.text(.055,.12,footer,fontsize=9,color='#45536a');fig.text(.055,.075,limitations,fontsize=9,color='#45536a')
    if metric=='sampled_peak_adapter_gib':fig.text(.055,.03,'Includes background allocations and reserved caches; configured physical capacity differs. Not process-only or instantaneous peak.',fontsize=9,color='#45536a')
    fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.26,wspace=.22);save(fig,stem)

fig,axes=plt.subplots(2,1,figsize=(13,7));fig.set_facecolor('#f7f9fc')
matrix=np.array([[receipt['actual_dispatch_by_cohort'][key]['mtp_requests'] for key in cohorts] for receipt in (old,proof)])
assert matrix.shape==(2,8) and np.all((matrix>=0)&(matrix<=8))
batch_matrix=np.array([[8/receipt['actual_dispatch_by_cohort'][key]['groups'] for key in cohorts] for receipt in (old,proof)])
assert np.all((batch_matrix>=1)&(batch_matrix<=4))
for ax,values,maximum,title in zip(axes,(matrix,batch_matrix),(8,4),('Verified MTP requests / 8 measured requests','Average real request rows per dispatch · physical capacity 4')):
    ax.imshow(values,cmap='Blues',vmin=0,vmax=maximum,aspect='auto')
    for row in range(2):
        for col in range(8):
            label=f'{int(values[row,col])}/8' if maximum==8 else f'{values[row,col]:.2f}'
            ax.text(col,row,label,ha='center',va='center',color='white' if values[row,col]>maximum/2 else '#1f3550',fontweight='bold')
    ax.set_title(title,loc='left',fontsize=11,pad=8);ax.set_xticks(range(8),[case.title()+' C'+str(c) for case in cases for c in concurrency]);ax.set_yticks([0,1],['Register off','Register128']);ax.tick_params(length=0);ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True);ax.grid(which='minor',color='white',linewidth=2)
fig.suptitle('Actual MTP and batch occupancy',x=.055,y=.97,ha='left',fontsize=18,fontweight='bold');fig.text(.055,.9,'Saved worker events and dispatch membership; client concurrency alone does not establish real batch occupancy.',fontsize=10,color='#45536a')
fig.text(.055,.105,'MTP cells count eligible requests; remaining requests used greedy decoding. Batch averages exclude padding; no outputs were omitted.',fontsize=9,color='#45536a');fig.text(.055,.055,limitations,fontsize=9,color='#45536a')
fig.subplots_adjust(left=.14,right=.98,top=.81,bottom=.21,hspace=.55);save(fig,'Garnet-Register128-MTP-HTTP-Dispatch')

provenance={'status':'verified_pilot_charts_generated','proof_sha256':sha(proof_path.read_bytes()),'parent_proof_sha256':sha(parent_path.read_bytes()),'generator_sha256':sha(Path(__file__).read_bytes()),
    'archives':archives,'gpu':gpu,'all_cohorts_included':cohorts,'series':data,'actual_mtp_requests':matrix.tolist(),'real_rows_per_dispatch':batch_matrix.tolist(),'two_x_established':False,
    'scope':footer+' '+limitations,'artifacts':{p.name:sha(p.read_bytes()) for p in artifacts}}
p=destination/'Garnet-Register128-MTP-HTTP-Chart-Provenance.json';p.write_text(json.dumps(provenance,indent=2),encoding='utf8');print(json.dumps({'charts':len(artifacts)//2,'files':[p.name for p in artifacts]+[p.name]}))
