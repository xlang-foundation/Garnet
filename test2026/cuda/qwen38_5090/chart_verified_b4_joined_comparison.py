"""Chart joined/unjoined B4 pilots only after both raw archives pass replay."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);sha=lambda b:hashlib.sha256(b).hexdigest()
names=['b4-host-joined-mtp-http-pilot-attempt1','batch4-t4-mtp-host-embedding-http-pilot-attempt1'];proofs=[];inputs={}
for name,status in zip(names,['b4_host_joined_mtp_http_pilot_replayed','batch4_t4_mtp_host_embedding_http_pilot_independently_verified']):
 raw=(out/(name+'-verified.json')).read_bytes();d=json.loads(raw);assert d['status']==status
 assert sha((out/(name+'-evidence.zip')).read_bytes())==d['archive_sha256']
 assert d['physical_batch']==4 and len(d['run']['cohorts'])==8 and not d['two_x_established']
 inputs[name]={'verified_sha256':sha(raw),'archive_sha256':d['archive_sha256']};proofs.append(d)
joined,original=proofs;assert joined['joined_mlp_projections']
assert [r['run_id'] for r in joined['references']]==[r['run_id'] for r in original['references']]
assert [r['engine'] for r in joined['references']]==['vllm','sglang']
for key,cohort in joined['run']['cohorts'].items():
 assert cohort['requests']==original['run']['cohorts'][key]['requests'],key
runs=[joined['run'],original['run']]+joined['references']
labels=['Garnet B4 joined MTP3','Garnet B4 unjoined MTP3','vLLM ordinary','SGLang ordinary'];colors=['#1c75b5','#8d60ab','#db9c28','#31856b'];cs=[1,2,4,8]
metrics=[('ttft_median_ms','Median TTFT / ms'),('decode_median_tokens_per_second','Median request decode / tok/s'),('output_tokens_per_second','Aggregate output / tok/s')]
fig,axes=plt.subplots(4,2,figsize=(13,13));fig.set_facecolor('#f7f9fc')
for col,case in enumerate(['text','image']):
 for row,(metric,title) in enumerate(metrics):
  ax=axes[row,col]
  for run,label,color in zip(runs,labels,colors):ax.plot(cs,[run['cohorts'][f'{case}-c{c}']['metrics'][metric] for c in cs],'o-',color=color,label=label,linewidth=2)
  ax.set_ylim(bottom=0);ax.set_title(case.title()+' | '+title,loc='left',fontsize=11)
  if row==0 and col==0:fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.915),ncol=2,fontsize=10)
 ax=axes[3,col]
 for d,label,color in zip(proofs,labels,colors):
  rows={c['concurrency']:c for c in d['raw_cohorts'] if c['case']==case}
  ax.plot(cs,[rows[c]['sampled_peak_adapter_used_bytes']/2**30 for c in cs],'o-',color=color,label=label)
 ax.set_ylim(0,32);ax.set_title(case.title()+' | Garnet sampled adapter peak / GiB',loc='left',fontsize=11)
 for ax in axes[:,col]:ax.set_xticks(cs);ax.set_xlabel('Client concurrency');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Joined projections | matched batch-4 pilot comparisons',x=.07,ha='left',fontsize=18,fontweight='bold')
fig.text(.07,.94,'Qwen3.8-27B VLM | RTX 5090 32GB | all eight measured cohorts retained',fontsize=11)
fig.text(.07,.08,'Same request bytes: 41 text / 104 image prompt tokens, 128 outputs, 8 requests/cohort; capacity 4096.',fontsize=9)
fig.text(.07,.06,'Full-four equal-limit groups use MTP3; partial/unequal groups use greedy. Client concurrency does not guarantee MTP.',fontsize=9)
fig.text(.07,.02,'Separate single pilots; no error bars or causal speedup claim. Clean repeats, tuned references, actual 4K and broad quality pending.',fontsize=9)
fig.text(.07,.04,'Image C4 MTP requests: joined '+str(joined['actual_dispatch_by_cohort']['image-c4']['mtp_requests'])+'/8; unjoined '+str(original['actual_dispatch_by_cohort']['image-c4']['mtp_requests'])+'/8. C8: both cases and both profiles 8/8.',fontsize=9)
fig.subplots_adjust(left=.08,right=.96,top=.875,bottom=.14,hspace=.65,wspace=.25)
for ext in ['png','svg']:fig.savefig(out/('Garnet-B4-Joined-vs-Unjoined-Comparison.'+ext),dpi=160,facecolor=fig.get_facecolor())
(out/'Garnet-B4-Joined-Chart-Provenance.json').write_text(json.dumps({'inputs':inputs,'generator_sha256':sha(Path(__file__).read_bytes()),'all_cohorts_retained':True,'causal_speedup_established':False},indent=2))
