# SPDX-License-Identifier: Apache-2.0
"""Check every declared SDK, original numerical and new internal-store gate."""
import hashlib
import json
from pathlib import Path
import re
import sys

root=Path(sys.argv[1]);stem='internal-store-numerical-v1'
def read(name):return json.loads((root/name).read_text(encoding='utf-8-sig'))
identity=read(stem+'-before-validation-identity.json');native=identity['garnet_binary_sha256']
assert read(stem+'-controller-exit.json')['exit_code']==0
records=[]
def gate(tag,kind='plain'):
    status=read(tag+'-exit.json');data=read(tag+'.json')
    assert status['exit_code']==0 and status['garnet_dll_sha256'].lower()==native and data['all_passed'],tag
    if kind!='plain':
        assert status['kernel_filter']=='all'
        log=(root/(tag+'.log')).read_text(encoding='utf-8-sig')
        pattern=r'RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)' if kind=='racecheck' else r'ERROR SUMMARY:\s+0 errors'
        assert re.search(pattern,log),tag
    records.append({'tag':tag,'diagnostic':kind,'exit_code':0,
                    'sha256':hashlib.sha256((root/(tag+'.json')).read_bytes()).hexdigest()})
    return data
for kind in ('plain','memcheck','racecheck','synccheck'):
    data=gate(stem+'-sdk-'+kind,kind)
    assert data['output_checks']==50 and data['threaded_cloned_outputs']==24
    assert data['caller_pool_drop_partial_graph_close_io_close_and_last_output_owner']

modes={}
for mode in ('default','joined','shared','internal'):
    join=mode!='default';share=mode in ('shared','internal');internal=mode=='internal'
    profiles=((8,4),(4,4),(8,16),(4,16)) if mode=='default' else ((1,4),(2,4),(4,4),(8,4))
    mode_records=[]
    for batch,chunk in profiles:
        tag=f'{stem}-{mode}-verifier-b{batch}-c{chunk}';data=gate(tag)
        assert data['batch']==batch and data['chunk']==chunk
        assert data['join_decode_input_projections']==join and data['shared_activation_storage']==share
        assert all(data[k] for k in ('packed_program','prefix_replay','native_prefix_commit','shared_output_storage',
                                    'deferred_state_commit','register_delta_128','flat_concat','resident_controls','serial_target_graph_reuse'))
        assert data['invalid_prefix_token_control_capacity_and_close_rollback'] and all(data['deferred_checks'].values())
        assert len(data['checks'])==6
        mode_records.append({'tag':tag,'prefix_patterns':len(data['checks'])})
    for batch in (1,2,4,8):
        for prefill in (32,64):
            tag=f'{stem}-{mode}-session-b{batch}-p{prefill}-d4';data=gate(tag)
            assert data['batch']==batch and data['prefill_chunk']==prefill and data['chunk']==4
            assert data['prompt_tokens']==67 and data['draft_depth']==3 and len(data['records'])==24
            assert data['join_decode_input_projections']==join and data['shared_activation_storage']==share
            assert all(data[k] for k in ('native_prefix_commit','reuse_verified_head','shared_output_storage','packed_program',
                                        'prefix_replay','deferred_state_commit','register_delta_128','flat_concat','resident_controls',
                                        'serial_graph_reuse','serial_target_graph_reuse'))
            bodies=data['target_memory']['graphs'][:-1]
            assert sum(g['internal_packed_stores'] for g in bodies)==(2*len(bodies) if internal else 0)
            assert sum(g['internal_packed_root_fills'] for g in bodies)==(len(bodies) if internal else 0)
            assert all(not g['network_output_aliases_input'] for g in data['target_memory']['graphs'])
            mode_records.append({'tag':tag,'session_cases':len(data['records'])})
    assert len(mode_records)==12;modes[mode]=mode_records

full=gate(stem+'-fullstate-equivalence-memcheck','memcheck')
assert [r['batch'] for r in full['records']]==[1,2,4,8]
assert all(r['same_all_full_finite_states_and_ids'] and len(r['snapshots'])==9 for r in full['records'])
result={'status':'ALL_SDK4_ORIGINAL36_INTERNAL12_FULLSTATE_MEMCHECK_PASSED',
        'garnet_dll_sha256':native,'sdk_modes':4,'sdk_output_checks_each':50,
        'original_numerical_cases':36,'internal_numerical_cases':12,'modes':modes,'sources':records,
        'scope':'Synthetic native numerical/state/rollback/stop/fault/ownership controls, including explicit oracle proposals. Not pretrained correctness, benchmark timing or independent quality acceptance.'}
destination=root/'internal-packed-store-numerical-summary.json'
destination.write_text(json.dumps(result,indent=2),encoding='utf-8');print(destination)
