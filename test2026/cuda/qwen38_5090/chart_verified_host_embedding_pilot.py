"""Plot latest verified host-embedding pilot without dropping slow cohorts."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);name='batch-mtp-host-embedding-http-pilot-attempt1'
raw=(out/(name+'-verified.json')).read_bytes();d=json.loads(raw)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert d['status']=='batch_mtp_host_embedding_http_pilot_independently_verified'
assert sha((out/(name+'-evidence.zip')).read_bytes())==d['archive_sha256']
assert d['clean_repeat_required'] and not d['two_x_established']
assert [r['engine'] for r in d['references']]==['vllm','sglang']
fig,axes=plt.subplots(4,2,figsize=(13,13));fig.set_facecolor('#f7f9fc')
metrics=[('ttft_median_ms','Median TTFT · ms ↓'),('decode_median_tokens_per_second','Median request decode · tok/s ↑'),('output_tokens_per_second','Aggregate output · tok/s ↑')]
cs=[1,2,4,8];runs=[d['run']]+d['references'];labels=['Garnet B2 host MTP3','vLLM ordinary','SGLang ordinary'];colors=['#2576b8','#df9d28','#388c72']
for col,case in enumerate(['text','image']):
 for row,(metric,title) in enumerate(metrics):
  ax=axes[row,col]
  for run,label,color in zip(runs,labels,colors):
   vals=[run['cohorts'][f'{case}-c{c}']['metrics'][metric] for c in cs]
   ax.plot(cs,vals,'o-',label=label,color=color,linewidth=2)
   ax.annotate(f'{vals[-1]:.1f}',(8,vals[-1]),xytext=(6,[0,7,-7][labels.index(label)]),textcoords='offset points',fontsize=9,color=color)
  ax.set_title(case.title()+' · '+title,loc='left',fontsize=11)
  ax.set_ylim(bottom=0);ax.set_xlim(.7,9.3)
  if row==0 and col==0:fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.915),ncol=3,fontsize=10)
 ax=axes[3,col];coh=[next(c for c in d['raw_cohorts'] if c['case']==case and c['concurrency']==n) for n in cs]
 ax.plot(cs,[c['sampled_peak_adapter_used_bytes']/2**30 for c in coh],'o-',color=colors[0],label='Garnet sampled adapter peak')
 ax.set_ylim(0,32);ax.set_title(case.title()+' · sampled GPU memory / GiB',loc='left',fontsize=11);ax.legend(fontsize=8)
 for ax in axes[:,col]:
  ax.set_xticks(cs);ax.set_xlabel('Client concurrency');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Latest verified Garnet host-embedding pilot vs references',x=.07,ha='left',fontsize=18,fontweight='bold')
fig.text(.07,.94,'Qwen3.8-27B · RTX 5090 32GB · physical Garnet batch 2 · all measured cohorts retained',fontsize=11)
fig.text(.07,.065,'41 text / 104 image prompt tokens, 128 outputs, 8 requests/cohort. 4096 capacity; actual 4K prompts pending.',fontsize=9)
fig.text(.07,.045,'Garnet singles use greedy; equal-limit pairs use MTP3. C2 dispatch is mixed. References are ordinary, untuned pilots.',fontsize=9)
fig.text(.07,.025,'Single pilot; no error bars. 22.3 s text-C4 / 14.6 s image-C2 sampler gaps, cause unknown. Clean repeats required; 2× not established.',fontsize=9)
fig.subplots_adjust(left=.08,right=.95,top=.88,bottom=.12,hspace=.65,wspace=.25)
for ext in ['png','svg']:fig.savefig(out/('Garnet-Latest-Host-Embedding-Comparison.'+ext),dpi=160,facecolor=fig.get_facecolor())
(out/'Garnet-Latest-Host-Embedding-Chart-Provenance.json').write_text(json.dumps({'verified_input_sha256':sha(raw),'archive_sha256':d['archive_sha256'],'generator_sha256':sha(Path(__file__).read_bytes()),'all_8_cohorts_retained':True,'two_x_established':False},indent=2))



