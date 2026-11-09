"""Add a plain vector4 comparator; preserve the three-method pilot unchanged."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
s=(R/'screen1.mm').read_text()
def repl(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
repl('root/"plan1.json"','root/"plan2.json"')
repl('std::array<id<MTLComputePipelineState>,4> psos','std::array<id<MTLComputePipelineState>,5> psos')
repl('for(int i=0;i<4;++i)','for(int i=0;i<5;++i)')
repl('i==1?(batch==1?@"verification_attention_context_vector4":@"verification_attention_context"):@"verification_gqa16_context"','i==1?(batch==1?@"verification_attention_context_vector4":@"verification_attention_context"):i==4?@"verification_attention_context_vector4":@"verification_gqa16_context"')
repl('if(i>=2){auto fc=','if(i==2||i==3){auto fc=')
repl('i>=2?(i==2?6144:12288):0','i==2?6144:i==3?12288:0')
repl('if(method){[e setThreadgroupMemoryLength:','if(method==1||method==2){[e setThreadgroupMemoryLength:')
repl('size_t count=batch*4*4096/(batch==1?4:1),threads=batch==1?64:','size_t count=batch*4*4096/((batch==1||method==3)?4:1),threads=(batch==1||method==3)?64:')
repl('std::array<std::vector<uint32_t>,3> results','std::array<std::vector<uint32_t>,4> results')
repl('for(int order=0;order<3;++order){int method=cohort==1?order:2-order;','for(int order=0;order<4;++order){int method=cohort==1?order:3-order;')
repl('for(int m=1;m<3;++m)','for(int m=1;m<4;++m)')
repl('for(int order=0;order<3;++order){int method=(order+(round+4)%3)%3;if(cohort==2)method=2-method;','for(int order=0;order<4;++order){int method=(order+(round+4)%4)%4;if(cohort==2)method=3-method;')
(R/'screen2.mm').write_text(s)
plan=json.loads((R/'plan1.json').read_text());plan['methods']=['current','tile32','tile64','plain-vector4'];plan['reason_for_second_version']='Separate shared-memory GQA reuse from plain four-column vectorization, especially for the currently scalar native B4 path.'
(R/'plan2.json').write_text(json.dumps(plan,indent=2)+'\n')
s=(R/'run1.py').read_text().replace("R/'plan1.json'","R/'plan2.json'").replace('preflight1.json','preflight2.json').replace('postflight1.json','postflight2.json').replace('build1','build2').replace('screen1','screen2').replace('execution-bindings1.json','execution-bindings2.json').replace('run1-cohort','run2-cohort').replace("'run1.py'","'run2.py'").replace("'plan1.json'","'plan2.json'")
# Keep all four screen executable names in the idle inventory.
s=s.replace("['screen2','screen2','screen3','screen4']","['screen1','screen2','screen3','screen4']")
s=s.replace("'value1.metal','helpers1.inc','main1.inc'","'value1.metal','helpers1.inc','main1.inc','prepare2.py'")
(R/'run2.py').write_text(s)
