# SPDX-License-Identifier: Apache-2.0
"""Report bounded real-image preservation controls, not independent quality."""
import hashlib
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]);stem='internal-store-pretrained-v1'
def read(name):return json.loads((root/name).read_text(encoding='utf-8-sig'))
assert read(stem+'-controller-exit.json')['exit_code']==0
identity=read(stem+'-before-execution-identity.json');native=identity['garnet_binary_sha256']
cases=[]
for size,batch in (('2b',1),('4b',8)):
    tag=f'{stem}-{size}-b{batch}';data=read(tag+'.json');status=read(tag+'-exit.json');prior=read(tag+'-cross-prior-state-control.json')
    assert status['exit_code']==0 and status['garnet_dll_sha256'].lower()==native
    assert data['same_recorded_sequences'] and data['same_ordinary_outputs'] and data['all_state_hashes_equal']
    assert data['batch']==batch and len(data['trials'])==2 and len(data['state_hashes'])==2
    assert data['native_profile']['internalPackedStores'] and not data['ordinary_profile']['internalPackedStores']
    assert data['state_hash_method']=='bounded_native_finite_bytes'
    assert prior['all_passed'] and all(prior['checks'].values())
    assert prior['declared_profile_transition']=='internalPackedStores False to True only'
    hashes=data['ordinary_state_hashes']
    cases.append({'model':'Qwen3.5-2B' if size=='2b' else 'Qwen3.5-4B','batch':batch,
                  'canonical_fp32_words_per_request':hashes['canonical_state']['fp32_words'],
                  'hidden_fp32_words_per_request':hashes['last_hidden']['fp32_words'],
                  'ordinary_hashes':hashes,'two_repeated_requests_and_ordinary_exact':True,
                  'prior_complete_states_and_outputs_exact':True,'prior_checks':prior['checks'],
                  'raw_result':tag+'.json','raw_sha256':hashlib.sha256((root/(tag+'.json')).read_bytes()).hexdigest()})
result={'status':'BOTH_PRETRAINED_ORDINARY_PRIOR_REPEATED_EXACT_CONTROLS_PASSED','garnet_dll_sha256':native,'cases':cases,
        'scope':'Two repeated native-prefix requests then a separately loaded ordinary non-MTP target per model; one real-image group each. Every finite canonical-state/hidden word hashed. Exact prior profile/output/stop preservation with only declared internal-store flag change.',
        'limits':['Not a full28 timing cohort or independent math/vision/quality oracle.',
                  'State hashing and cold/repeated calls invalidate primary latency claims.',
                  'All optimized batch/profile and broader teacher/capacity coverage remain in the full goal.']}
output=root/'internal-packed-store-pretrained-summary.json';output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(output)
