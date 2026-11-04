"""Audit complete frozen MTP coding IDs, native transactions and shared weights.

Run after inference and sandbox evaluation have both finished.
"""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root):
    root = Path(root)
    checks = 0

    def require(condition, label):
        nonlocal checks
        if not condition:
            raise ValueError(label)
        checks += 1

    read = lambda p: json.loads(Path(p).read_text())
    for file in (root/'provenance.json', root/'engine/provenance.json'):
        provenance = read(file)
        for name, sha in provenance['sha256'].items():
            require(digest(name) == sha, 'Changed qualified input/source/library: '+name)
    require(read(root/'engine/provenance.json')['asset_manifest_sha256'] ==
            'db7aecc1a6bd8911e5737c0dc3f13089f048638669177f8c8dc331df0a6ad198', 'Target asset manifest')
    result = read(root/'engine/result.json')
    fixture = read(root/'coding.json')
    evaluation = read(root/'evaluation.json')
    require(result['passed'] is True and len(result['tasks']) == len(fixture['plans']) == 8, 'Complete eight tasks required')
    require(evaluation['passed'] == evaluation['total'] == 8, 'Sandbox coding gate')
    require(evaluation['suite_sha256'] == '675ea667c357f361ce39e3863b61193dc6b32f4a229eb7d29a17803b75408786', 'Frozen coding suite')
    require(all(row['status'] == 'passed' for row in evaluation['results']), 'Every sandbox task must pass')
    tasks = []
    for plan, task in zip(fixture['plans'], result['tasks']):
        require(task['id'] == plan['id'] and task['prompt_ids'] == plan['prompt_ids'], 'Frozen task identity and prompt')
        require(task['generated_ids'] == plan['reference_ids'] and task['full_token_match'] is True, 'Complete frozen output IDs')
        response_file = root/'responses'/plan['id']/'request-0.result.json'
        response = read(response_file)
        require(response['text'] == plan['reference_text'] and task['text_match'] is True, 'Complete frozen output text')
        require(response['status'] == 'completed' and response['finish_reason'] == 'stop', 'Nontruncated completion')
        require(response['generated_ids'] == task['generated_ids'] and response['generated_tokens'] == len(task['generated_ids']), 'Saved response token accounting')
        require(response['prompt_tokens'] == len(plan['prompt_ids']), 'Actual input token count')
        sandbox = [row for row in evaluation['results'] if row['task'] == plan['id']]
        require(len(sandbox) == 1 and sandbox[0]['response_sha256'] == digest(response_file), 'Evaluated exact saved response')
        prefill = task['prefill']
        chunks = (len(plan['prompt_ids'])+127)//128
        require(prefill['committed'] == prefill['tentative'] == [len(plan['prompt_ids'])] and prefill['pending'] is False, 'Actual ragged prefill history')
        require(prefill['profile_chunk'] == 128 and prefill['teacher_calls'] == chunks and prefill['head_calls'] == 1 and prefill['validation_only_calls'] == chunks-1, 'Prepared wide teacher and skipped intermediate heads')
        generated = task['generated_ids'][:1]
        proposed = accepted = 0
        for transaction in task['transactions']:
            stats = transaction['stats']
            consumed = len(transaction['emitted'])
            remaining = plan['max_output']-len(generated)
            require(transaction['remaining_before'] == remaining and 1 <= consumed <= min(4, remaining), 'Output budget')
            require(stats['consumed'] == [consumed] and 1 <= stats['submitted_lengths'][0] <= min(4, remaining), 'Native consumption/submission')
            require(consumed <= stats['submitted_lengths'][0] and stats['proposals'][0] == generated[-1], 'Authoritative known input')
            require(stats['proposed_drafts'] == stats['submitted_lengths'][0]-1 and stats['accepted_drafts'] == consumed-1, 'Draft acceptance accounting')
            require(stats['draft_profile_chunk'] == 1 and stats['correction_profile_chunk'] == 4, 'Prepared native draft/correction profiles')
            generated.extend(transaction['emitted'])
            final = len(generated) == len(task['generated_ids'])
            require(transaction['eos'] is final and stats['auxiliary_validation_calls'] == int(final), 'EOS/final cache-only correction')
            proposed += stats['proposed_drafts']
            accepted += stats['accepted_drafts']
        require(generated == task['generated_ids'] and len(generated) <= plan['max_output'], 'Every native emitted ID reconstructed')
        memory = task['memory']
        require(memory['metal_allocated_bytes'] <= memory['metal_recommended_working_set_bytes']-256*1024*1024, 'Task allocation snapshot guard')
        tasks.append(dict(id=plan['id'], prompt_tokens=len(plan['prompt_ids']), output_tokens=len(generated),
                          transactions=len(task['transactions']), proposed_drafts=proposed, accepted_drafts=accepted))
    inventory = read(result['inventory'])
    weights = {}
    for ref in inventory['references']:
        if ref['tag'].startswith('auxiliary_') and ref['role'] == 'weight_record':
            weights.setdefault(ref['tag'], {}).setdefault(ref['name'], []).append(ref['buffer_id'])
    require(set(weights) == {'auxiliary_0', 'auxiliary_1', 'auxiliary_2'}, 'Three explicit auxiliary profiles')
    require(len(weights['auxiliary_0']) == 15 and sum(name.endswith('|BF16') for name in weights['auxiliary_0']) == 8, 'Eight BF16 matrices and seven norms')
    require(weights['auxiliary_0'] == weights['auxiliary_1'] == weights['auxiliary_2'], 'All fifteen auxiliary weight buffers shared')
    all_weights = {}
    for ref in inventory['references']:
        if ref['role'] == 'weight_record':
            all_weights.setdefault(ref['tag'], {}).setdefault(ref['name'], []).append(ref['buffer_id'])
    projections = [records for records in all_weights.values() if set(records) == {'lm_head.weight'}]
    require(len(projections) == 1 and len(projections[0]['lm_head.weight']) == 3, 'One projection-only affine vocabulary head')
    require(projections[0]['lm_head.weight'] == all_weights['target_head']['lm_head.weight'], 'Auxiliary projection shares all target head buffers')
    require(result['prepared_memory']['garnet_live_affine_weight_owners'] == 497, 'No duplicated target affine matrices')
    require(result['prepared_memory']['garnet_live_bf16_weight_owners'] == 8, 'No duplicated raw auxiliary matrices')
    require(result['prepared_memory']['metal_allocated_bytes'] <= result['prepared_memory']['metal_recommended_working_set_bytes']-256*1024*1024, 'Prepared allocation snapshot guard')
    closed = [json.loads(line.split(' ', 1)[1]) for line in (root.parent/(root.name+'.log')).read_text().splitlines()
              if line.startswith('native_mtp_coding_closed ')]
    require(len(closed) == 1, 'One completed native cleanup record')
    for name in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners', 'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        require(closed[0][name] == 0, 'Closed native owner: '+name)
    return dict(status='passed', checks=checks, tasks=tasks, closed_owners=closed[0],
                auditor_sha256=digest(__file__),
                result_sha256=digest(root/'engine/result.json'), evaluation_sha256=digest(root/'evaluation.json'),
                scope='B1 frozen coding/full IDs and native profile accounting; snapshots not peak, no model speed/batching/prefix admission')


if __name__ == '__main__':
    report = audit(sys.argv[1])
    with Path(sys.argv[2]).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
