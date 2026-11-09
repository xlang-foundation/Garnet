"""Retain independent causal/tokenizer validators while checking the new compiled fusion route."""
from pathlib import Path
R=Path(__file__).resolve().parent;N=R.parent/'mistral-gqa16-value-native1'
s=(N/'audit_bounded_value_common1.py').read_text();marker="     shared='native_mistral_verification_gqa16_values64_32_optional_v1' in data['backend']";assert s.count(marker)==1
s=s.replace(marker,"     fused='native_fused_q4_decode_rows4_default_v1' in data['backend']\n     require(data['fused_decode_enabled']==fused,'compiled fused route')\n     require(metrics.get('target_fused_q4_decode_blocks',0)==(32 if fused and len(proposal)==3 else 0),'actual fused gate/up route')\n     if fused:require(0<=metrics['target_fused_q4_decode_input_copies']<=metrics['target_fused_q4_decode_blocks'],'fused private input copies')\n"+marker)
(R/'audit_common1.py').write_text(s)
s=(N/'audit_bounded_qualification1.py').read_text().replace('from audit_bounded_value_common1 import validate','from audit_common1 import validate')
s=s.replace("data['fused_decode_enabled'] is False","data['fused_decode_enabled']==('native_fused_q4_decode_rows4_default_v1' in data['backend'])")
s=s.replace("data['prepared']['fused_decode_profiles']==0","(1<=data['prepared']['fused_decode_profiles']<=2 if data['fused_decode_enabled'] else data['prepared']['fused_decode_profiles']==0)")
(R/'audit_bounded1.py').write_text(s)
for n in ['audit_common1.py','audit_bounded1.py']:compile((R/n).read_text(),n,'exec')
