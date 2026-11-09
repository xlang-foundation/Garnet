"""Keep legacy route tests explicit; qualify shared values independently."""
from pathlib import Path
R=Path(__file__).resolve().parent;S=R/'source1';N=R.parent/'mistral-gqa16-score-native1/source1'
s=(N/'test2026/metal/mistral_verification_gqa_score_test.cpp').read_text()
def replace(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
replace('bool gqaOn=true;','bool gqaOn=true;\nbool valueOn=true;')
visible='size_t visible=0;for(size_t r=0;r<j.batch;++r)for(size_t t=0;t<j.chunk;++t)if(j.active[r*j.chunk+t])visible=std::max(visible,size_t(j.slots[r*j.chunk+t])+1);'
replace('    '+visible+'\n','')
replace('#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4\n    const bool vector=profiled&&allowVector;', '    '+visible+'''
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    const bool shared=valueOn&&(j.batch==1||j.batch==4)&&j.chunk==4&&j.heads==32&&j.kv==8&&j.dim==128&&j.capacity==4096&&!gated&&visible>=512;
#else
    const bool shared=false;
#endif
    Need(gpu.LastVerificationGqaValueDispatches()==size_t(shared),"Shared-value dispatch proof differs");
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    {bool rejected=false;try{gpu.VerificationSetGqaValuesEnabled(false);}catch(const std::logic_error&){rejected=true;}
     Need(rejected&&gpu.LastVerificationGqaValueDispatches()==size_t(shared),"Open/partial batch changed shared-value policy");++rejections;}
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
    const bool vector=profiled&&allowVector&&!shared;''')
replace('    gpu.Begin();Need(gpu.LastVerificationAttentionDispatches()==0,','    gpu.Begin();Need(gpu.LastVerificationGqaValueDispatches()==0,"Begin retained shared-value counter");Need(gpu.LastVerificationAttentionDispatches()==0,')
# The ownership fixture toggles only the new value policy; QK stays identical.
start=s.index('void Ownership(');end=s.index('void InvalidGqa(',start)
owner=s[start:end].replace('void Ownership(','void ValueOwnership(').replace('GqaRoute(', 'ValueRoute(').replace('GQA16_SCORES','GQA16_VALUES').replace('LastVerificationGqaScoreDispatches','LastVerificationGqaValueDispatches').replace('VerificationSetGqaScoresEnabled','VerificationSetGqaValuesEnabled').replace('"GQA16','"Shared-value')
value_route='''void ValueRoute(MetalContext& gpu,bool enabled) {
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    gpu.VerificationSetGqaValuesEnabled(enabled);
#else
    (void)gpu;
#endif
    valueOn=enabled;
}
'''
pos=s.index('void InvalidGqa(');s=s[:pos]+value_route+owner+s[pos:]
replace('     Invalid(gpu,foreign);InvalidGqa(gpu);', '''     for(size_t batch:{size_t(1),size_t(4)})for(bool flush:{false,true}) {
         ValueOwnership(gpu,foreign,batch,2048,4,flush);ValueOwnership(gpu,foreign,batch,4095,1,flush);
     }
     for(size_t batch:{size_t(1),size_t(4)}) {
         ValueRoute(gpu,false);Case(gpu,Job(batch,4,32,8,128,4096,2048,0),false,0);
         ValueRoute(gpu,true);Case(gpu,Job(batch,4,32,8,128,4096,2048,0),false,2);
         Case(gpu,Job(batch,4,32,8,128,4096,508,0),false,0);
     }
     Invalid(gpu,foreign);InvalidGqa(gpu);''')
replace('    {MetalContext gpu,foreign;','''    {MetalContext gpu,foreign;
#ifndef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
     Reject([&]{gpu.VerificationSetGqaValuesEnabled(true);},"DefaultOFF build accepted shared values");
#endif''')
s=s.replace('"Mistral GQA16 scores PASS cases="','"Mistral GQA16 values PASS cases="').replace('"Mistral GQA16 FAILED "','"Mistral GQA16 values FAILED "')
(S/'test2026/metal/mistral_verification_gqa_value_test.cpp').write_text(s)
for name in ['mistral_verification_layout_test.cpp','mistral_verification_split_score_test.cpp','mistral_verification_gqa_score_test.cpp']:
 p=S/'test2026/metal'/name;s=p.read_text()
 anchor='    {MetalContext gpu,foreign;' if name.endswith('gqa_score_test.cpp') else '        MetalContext gpu,other;'
 assert s.count(anchor)==1
 s=s.replace(anchor,anchor+'\n#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES\n        gpu.VerificationSetGqaValuesEnabled(false);\n#endif')
 p.write_text(s)
print('Prepared independent native shared-value route, fallback and ownership tests.')
