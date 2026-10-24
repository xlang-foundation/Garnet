"""All-profile component ratios; never conflate with model/reference benchmarks."""
from pathlib import Path
import hashlib,json,math,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path(sys.argv[1]);name='attention-kv-delta-cached-attempt1'
proof_raw=(out/(name+'-verified.json')).read_bytes();proof=json.loads(proof_raw)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert proof['status']=='attention_kv_delta_cached_replayed' and proof['raw_context_arrays']==216 and proof['raw_prefix_count_vectors']==72
assert sha((out/(name+'-evidence.zip')).read_bytes())==proof['archive_sha256']
rows=proof['component_event_ratios'];assert len(rows)==12 and [r['case'] for r in rows]==list(range(12))
keys=['baseline_ms','compact_ms','tiled_ms','split_ms','overlay_commit_ms','cached_compact_commit_ms','cached_legacy_commit_ms','hybrid_commit_ms']
labels=['Legacy\nfull-cache','Compact\nfull-cache','Tiled\nfull-cache','Split\nfull-cache','Original\nKV overlay','Cached\ncompact','Cached\nlegacy','Hybrid\n512']
ratios=np.array([[r['median_ms']['baseline_ms']/r['median_ms'][k] for k in keys] for r in rows])
assert np.isfinite(ratios).all() and (ratios>0).all()
row_labels=[]
for r in rows:
 if r['pattern']==0:label='All active | prefix parameter '+str(r['prefix_parameter'])
 elif r['pattern']==1:label='Three rows | alternating active tokens'
 elif r['pattern']==2:label='All inactive | control only'
 else:label='Mixed row prefixes / masks'
 row_labels.append(label)
fig,ax=plt.subplots(figsize=(14,9));fig.set_facecolor('#f7f9fc')
plot=ax.imshow(np.log2(ratios),cmap='RdYlBu',vmin=-3,vmax=3,aspect='auto')
ax.set_xticks(range(8),labels);ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=8)
ax.set_yticks(range(12),row_labels)
for i in range(12):
 for j in range(8):ax.text(j,i,f'{ratios[i,j]:.2f}x',ha='center',va='center',fontsize=10,color='white' if abs(math.log2(ratios[i,j]))>2.4 else '#162438')
ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,12,1),minor=True)
ax.grid(which='minor',color='white',linewidth=1.3);ax.tick_params(which='minor',length=0)
for s in ax.spines.values():s.set_visible(False)
bar=fig.colorbar(plot,ax=ax,pad=.03,fraction=.03)
bar.set_ticks([-3,-2,-1,0,1,2,3],labels=['<=0.125x','0.25x','0.5x','1x','2x','4x','>=8x'])
bar.set_label('Legacy full-cache median / variant median')
fig.suptitle('KV cache update study | synthetic component ratios',x=.04,ha='left',fontsize=18,fontweight='bold')
fig.text(.04,.92,'RTX 5090 | B4 / T4 | FP32 | higher ratios are faster within this component',fontsize=11)
fig.text(.04,.115,'Left four columns return full K/V caches. Right four compute readonly overlays and explicitly commit accepted slots in place.',fontsize=10)
fig.text(.04,.087,'Different output / ownership contracts; fixed slots and common initialized cache values. Six event samples per variant, 20 graph launches each.',fontsize=9)
fig.text(.04,.059,'All twelve profiles and eight variants retained, including regressions. All-inactive results are control checks, not user-serving speedups.',fontsize=9)
fig.text(.04,.031,'Not full-model, actual 4K prompt, HTTP, vLLM/SGLang, peak-VRAM or 2x-target evidence. Timing variation requires controlled repeats.',fontsize=9)
fig.subplots_adjust(left=.31,right=.92,top=.82,bottom=.17)
for ext in ('png','svg'):fig.savefig(out/('Garnet-KV-Overlay-Component-Study.'+ext),dpi=160,facecolor=fig.get_facecolor())
(out/'Garnet-KV-Overlay-Component-Provenance.json').write_text(json.dumps({'archive_sha256':proof['archive_sha256'],'proof_sha256':sha(proof_raw),'generator_sha256':sha(Path(__file__).read_bytes()),'all_profiles':12,'all_variants':8,'model_performance_admitted':False},indent=2),encoding='utf8')
