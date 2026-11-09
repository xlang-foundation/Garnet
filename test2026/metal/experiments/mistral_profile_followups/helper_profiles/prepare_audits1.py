"""Adapt causal validators to the declared helper upper bound and native token-count witnesses."""
from pathlib import Path
R=Path(__file__).resolve().parent;V=R.parent/'mistral-fused-verify-gqa-profile1';s=(V/'audit_common1.py').read_text()
s=s.replace("inputs=data['inputs'];require", "cap=data['helper_prefill_maximum'];require(cap in (16,32,64,128,256) and data['prepared']['helper_prefill_maximum']==cap,'helper profile bound')\n inputs=data['inputs'];require")
s=s.replace("data['prepared']['draft_graphs']==draft_layers*4", "data['prepared']['draft_graphs']==draft_layers*(cap.bit_length()-1)")
s=s.replace("all(c in (1,2,4,8,16) for c in chunks)", "all(c in (1,2,4,8,16,32,64,128,256) and c<=cap for c in chunks)")
s=s.replace('      width=16\n', '      width=cap\n').replace('while width<remaining_input and width<16:width*=2','while width<remaining_input and width<cap:width*=2')
marker="     require(chunks==wanted_chunks,'wrong resynchronization profile policy')";assert s.count(marker)==1
s=s.replace(marker,marker+'''
     samples=trace['prefill_samples']
     if samples:
      require(len(samples)==len(chunks),'missing native prefill witnesses');cursor=trace['cached_tokens']
      for width,sample in zip(chunks,samples):
       actual=min(width,len(prompt)-cursor);require(sample['width']==width and sample['actual']==actual and sample['offset']==cursor and sample['tracked']==cursor+actual,'native helper token count/mask/profile witness')
       require(duration(sample['wall_seconds']) and duration(sample['gpu_ms']) and type(sample['mpp']) is int and sample['mpp']>=0 and type(sample['indexed']) is int and sample['indexed']>=0,'native helper route metrics');cursor+=actual
      require(cursor==len(prompt),'native helper prefill incomplete')''')
(R/'audit_common1.py').write_text(s)
s=(V/'audit_bounded1.py').read_text();s=s.replace("data['prepared']['draft_graphs']==96", "data['prepared']['draft_graphs']==24*(data['helper_prefill_maximum'].bit_length()-1)");(R/'audit_bounded1.py').write_text(s)
for n in ['audit_common1.py','audit_bounded1.py']:compile((R/n).read_text(),n,'exec')
