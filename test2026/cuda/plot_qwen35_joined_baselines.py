"""Plot the same equal-output subset for every listed engine per profile."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--activation-storage',action='store_true');p.add_argument('--expanded-conv',action='store_true');a=p.parse_args()
if a.expanded_conv and not a.activation_storage:p.error('Expanded convolution baseline requires activation study')
root=Path(__file__).resolve().parents[3]/'work/qwen35-cuda'
study='activation-arena' if a.activation_storage else 'joined-decode'
output='progress-activation-baselines' if a.activation_storage else 'progress-joined-baselines'
if a.expanded_conv:output+='-expanded-conv'
fig,axes=plt.subplots(1,2,figsize=(14,6),facecolor='#f7f9fc')
sources=[]
for ax,(size,batch) in zip(axes,(('2b',1),('4b',8))):
    extra='-expanded-conv' if a.expanded_conv and size=='2b' else ''
    path=root/f'packed-study-{size}-b{batch}-{study}-all-baselines{extra}.json'
    blob=path.read_bytes();data=json.loads(blob);values=data['median_group_ms']
    mtp=next(key for key in values if key.startswith('vllm-mtp3'))
    names=('garnet_b','lmstudio','vllm_serial',mtp)
    labels=('Garnet shared · repeated' if a.activation_storage else 'Garnet joined · repeated','LM Studio · saved','vLLM serial · saved','vLLM MTP3 · saved')
    if a.expanded_conv:
        mtp_names=[key for key in values if key.startswith('vllm-mtp3')]
        names=('garnet_b','lmstudio','vllm_serial',*mtp_names)
        labels=('Garnet shared · repeated','LM Studio · saved','vLLM serial · saved',
            *('vLLM MTP3 · BF16 conv' if 'convbfloat16' in key else 'vLLM MTP3 · FP32 conv' for key in mtp_names))
    walls=[values[key] for key in names]
    colors=('#159b83','#8293aa','#4688c4','#dd9c40','#8b6ab0')
    if a.expanded_conv:colors=('#159b83','#8293aa','#4688c4',*('#8b6ab0' if 'convbfloat16' in key else '#dd9c40' for key in mtp_names))
    bars=ax.barh(range(len(names)),walls,color=colors,height=.58)
    ax.invert_yaxis();ax.set_yticks(range(len(names)),labels,fontsize=10)
    for bar,value in zip(bars,walls):ax.text(value+max(walls)*.02,bar.get_y()+bar.get_height()/2,f'{value:.1f}',va='center',fontsize=10,weight='bold')
    target=data['half_fastest_baseline_ms'];ax.axvline(target,color='#a52b2b',linestyle='--',linewidth=1.5)
    ax.text(target,.99,f'2× target\n{target:.1f} ms',transform=ax.get_xaxis_transform(),ha='center',va='bottom',color='#a52b2b',fontsize=9)
    ratio=data['garnet_b_speed_ratio_vs_fastest']
    ax.set_title(f'Qwen3.5-{size.upper()} · batch {batch}\n{data["common_equal32_groups"]} common equal32 groups\nFastest baseline / Garnet: {ratio:.2f}×',loc='left',fontsize=11,weight='bold',pad=44)
    ax.set_xlabel('Full-request group median · ms (lower is better)',fontsize=10)
    ax.set_xlim(0,max(walls)*1.18);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.set_facecolor('#f7f9fc')
    for spine in ax.spines.values():spine.set_visible(False)
    sources.append({'file':path.name,'sha256':hashlib.sha256(blob).hexdigest()})
interference=[root/name for name in ('activation-paired-build-interference.json','activation-4b-build-interference.json') if a.activation_storage and (root/name).exists()]
for path in interference:sources.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
title='Garnet shared activations' if a.activation_storage else 'Garnet joined decode'
if interference:title+=' · PROVISIONAL'
fig.suptitle(title+(' vs local baselines' if a.expanded_conv else ' vs saved local baselines'),x=.04,ha='left',fontsize=16,weight='bold')
fig.text(.04,.905,'Same real-image hashes and equal32 output counts within each panel; one common subset per profile.',fontsize=10)
if interference:fig.text(.04,.105,'External build overlap and interruptions recorded: these latency results cannot support promotion or causal attribution.',fontsize=9,color='#a52b2b',weight='bold')
fig.text(.04,.07,('Garnet: prior repeated study. Comparators: one cohort each; the 2B BF16-conv profile was added in a later run.' if a.expanded_conv else 'Garnet: two repeats per group, then median across groups. Competitors: single historical cohorts; not rerun in this study.'),fontsize=9)
fig.text(.04,.04,'APIs, intermediate arithmetic and speculative settings differ. Independent quality and the 2× goal remain open.',fontsize=9,color='#a52b2b')
fig.subplots_adjust(left=.16,right=.97,bottom=.20,top=.69,wspace=.70)
for extension in ('png','svg'):fig.savefig(root/f'{output}.{extension}',dpi=160,facecolor=fig.get_facecolor())
(root/f'{output}-sources.json').write_text(json.dumps({'sources':sources,'provisional_due_to_external_build':bool(interference),'goal_2x_achieved':False,'independent_quality_accepted':False},indent=2),encoding='utf-8')
print(root/f'{output}.png')
