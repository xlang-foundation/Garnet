# SPDX-License-Identifier: Apache-2.0
"""Actual post-load owned allocation evidence, separate from process VRAM."""
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]);source=root/'internal-packed-store-real-allocation-summary.json'
data=json.loads(source.read_text(encoding='utf-8'))
assert data['both_profiles_have_context_and_owned_reduction']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,6.7))
colors=['#8c99a7','#20a58e']
for row,case in enumerate(data['records']):
    groups=[([case['control_owned_model_bytes'],case['candidate_owned_model_bytes']],2**30,'Owned allocations (GiB)'),
            ([case['control_shared_activation_bytes'],case['candidate_shared_activation_bytes']],2**20,'Decoder workspace (MiB)')]
    for col,(raw,unit,title) in enumerate(groups):
        ax=axes[row,col];values=[v/unit for v in raw]
        ax.bar([0,1],values,color=colors,width=.52,zorder=3)
        ax.set_xticks([0,1],['Control','Internal stores'])
        ax.set_title(f'{case["model"]} B{case["batch"]}: {title}',loc='left',fontsize=11,pad=9)
        ax.set_ylim(0,max(values)*1.29);ax.yaxis.grid(True,alpha=.22,zorder=0)
        for i,value in enumerate(values):ax.text(i,value+max(values)*.025,f'{value:.2f}',ha='center',va='bottom',fontweight='bold')
        saved=(raw[0]-raw[1])/1e6
        ax.text(.98,.95,f'{saved:,.1f} MB less ({100*(raw[0]-raw[1])/raw[0]:.1f}%)',
                transform=ax.transAxes,ha='right',va='top',fontsize=9,color='#146e60')
fig.suptitle('Qwen3.5 CUDA: internal packed store allocation study',fontsize=15,fontweight='bold',x=.06,ha='left',y=.98)
fig.text(.06,.935,'Four fresh sequential processes. Same BF16 weights and serving profiles; only internalPackedStores differs.',fontsize=10)
fig.text(.06,.08,'Post-load owned counters only; no inference. Workspace is included in owned allocations.\n'
         'Not warmed process VRAM, physical residency or lifetime peak. Driver/library/transient/graph costs may be absent.\n'
         'Pretrained correctness and latency still unverified for this feature. The 2x latency target remains unmet.',fontsize=9,color='#434a54',va='top')
fig.subplots_adjust(left=.075,right=.975,top=.855,bottom=.185,hspace=.48,wspace=.24)
base=root/'progress-internal-store-owned-allocations'
fig.savefig(str(base)+'.png',dpi=160,facecolor='white');fig.savefig(str(base)+'.svg',facecolor='white')
def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
Path(str(base)+'-sources.json').write_text(json.dumps({'scope':'Post-load owned counters, not process memory or performance.',
    'source':source.name,'source_sha256':digest(source),'plot_script_sha256':digest(Path(__file__)),
    'garnet_dll_sha256':data['garnet_dll_sha256']},indent=2),encoding='utf-8')
print(str(base)+'.png')
