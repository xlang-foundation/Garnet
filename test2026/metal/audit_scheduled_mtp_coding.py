"""Audit trained B2 MTP IDs, native row progress, shared weights and cleanup.

Args: run root, frozen B1 coding root, frozen HTTP admission root, NEW report.
This is correctness/memory evidence, never throughput qualification.
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


def audit(root, coding_root, http_root):
    root, coding_root, http_root = map(Path, (root, coding_root, http_root))
    read = lambda path: json.loads(Path(path).read_text())
    checks = 0

    def require(condition, label):
        nonlocal checks
        if not condition:
            raise ValueError(label)
        checks += 1

    for name in ('provenance.json', 'engine/provenance.json'):
        for path, wanted in read(root/name)['sha256'].items():
            require(digest(path) == wanted, 'Changed input/source/library: '+path)
    require(read(root/'engine/provenance.json')['asset_manifest_sha256'] ==
            'db7aecc1a6bd8911e5737c0dc3f13089f048638669177f8c8dc331df0a6ad198', 'Pinned target manifest')
    controls = read(coding_root/'coding.json')['plans']
    require(len(controls) == 8, 'Eight frozen coding controls')
    wanted = [dict(id=p['id'], prompt=p['prompt_ids'], max_output=p['max_output'],
                   expected=p['reference_ids'], coding=True) for p in controls]
    canonical = next(p for p in read(http_root/'sources/canonical.json') if p['id'] == 'p2048-c2-o128')
    workload = read(canonical['workload'])
    require(len(workload['requests']) == 2, 'Two frozen long controls')
    for row, request in enumerate(workload['requests']):
        wanted.append(dict(id='long-'+str(row), prompt=request['token_ids'], max_output=128,
                           expected=canonical['expected_ids'][row], coding=False))
    fixture = read(root/'tasks.json')
    require(fixture['plans'] == wanted, 'Frozen full prompts, outputs and budgets')
    result = read(root/'engine/result.json')
    require(result['passed'] is True and len(result['tasks']) == 10, 'Complete trained scheduled run')
    records = {task['id']: task for task in result['tasks']}
    require(len(records) == 10 and set(records) == {p['id'] for p in wanted}, 'Unique complete task identities')
    by_id = {i+1: p for i, p in enumerate(wanted)}
    generated = {i: p['expected'][:1] for i, p in by_id.items()}
    request_transactions = {i: [] for i in by_id}
    placements, offsets = {}, {i: 0 for i in by_id}
    for event in result['prefills']:
        identity = event['id']
        require(identity in by_id, 'Known native prefill request')
        plan = by_id[identity]
        row, epoch = event['row'], event['epoch']
        require(type(row) is int and 0 <= row < 2 and type(epoch) is int and epoch > 0, 'Physical prefill lease')
        placement = (row, epoch)
        require(placements.setdefault(identity, placement) == placement, 'Stable request lease')
        count, offset = event['tokens'], event['offset']
        require(type(count) is int and 1 <= count <= 128 and offset == offsets[identity], 'Contiguous bounded prefill')
        offsets[identity] += count
        require(offsets[identity] <= len(plan['prompt']) and event['final'] is (offsets[identity] == len(plan['prompt'])), 'Actual prefill completion')
    require(all(offsets[i] == len(p['prompt']) for i, p in by_id.items()), 'Every full prompt consumed')
    require(len(set(placements.values())) == 10, 'Every request owns a distinct row epoch')
    finished, last_row = set(), {}
    combined = refreshes = proposed = accepted = 0
    for frame in result['frames']:
        items, stats, aux = frame['requests'], frame['generation'], frame['auxiliary']
        require(1 <= len(items) <= 2 and len({p['id'] for p in items}) == len(items) and
                len({p['row'] for p in items}) == len(items), 'Distinct active decode rows')
        require(stats['recursive_draft_calls'] == 0 and stats['correction_profile_chunk'] == 2 and
                stats['draft_profile_chunk'] == 2 and aux['profile_chunk'] == 2, 'Prepared B2T2 auxiliary profile')
        require(aux['kv_inplace_updates'] == 2, 'Both auxiliary KV outputs use private caches')
        require(frame['hidden_pool_bindings'] == 64, 'All target verifier layers use private hidden pool')
        require(all(len(stats[k]) == n for k, n in (('proposals', 4), ('submitted_lengths', 2), ('consumed', 2))), 'B2 native statistics shape')
        active = {item['row'] for item in items}
        for row in set(range(2))-active:
            require(stats['submitted_lengths'][row] == stats['consumed'][row] == 0, 'Inactive rows consume nothing')
        frame_proposed = frame_accepted = 0
        for item in items:
            identity, row = item['id'], item['row']
            require(identity in by_id and identity not in finished, 'Live native request')
            require((row, item['epoch']) == placements[identity], 'Decode lease matches prefill')
            if row in last_row and last_row[row] != identity:
                previous = last_row[row]
                require(previous in finished and item['epoch'] > placements[previous][1], 'Row reuse only after completion with a newer epoch')
            last_row[row] = identity
            plan, ids = by_id[identity], generated[identity]
            remaining = plan['max_output']-len(ids)
            require(item['known'] == ids[-1] and item['remaining_before'] == remaining, 'Authoritative known token and budget')
            emitted = item['emitted']
            require(item['consumed'] == len(emitted) == stats['consumed'][row] and
                    1 <= len(emitted) <= min(2, remaining), 'Bounded emitted block')
            require(stats['submitted_lengths'][row] == min(2, remaining) and stats['proposals'][row*2] == ids[-1], 'Native submitted block and authoritative proposal')
            ids.extend(emitted)
            require(ids == plan['expected'][:len(ids)] and len(ids) <= len(plan['expected']), 'Every emitted reference token')
            done = len(ids) == len(plan['expected'])
            require(type(item['eos']) is bool and type(item['finished']) is bool and item['finished'] is done, 'Native completion boundary')
            require(item['finished'] is (item['eos'] or len(ids) == plan['max_output']), 'EOS or exact output budget')
            if done:
                finished.add(identity)
            request_transactions[identity].append(item)
            frame_proposed += stats['submitted_lengths'][row]-1
            frame_accepted += len(emitted)-1
        require(stats['proposed_drafts'] == frame_proposed and stats['accepted_drafts'] == frame_accepted, 'Draft acceptance accounting')
        combined += int(len(items) == 2)
        refreshes += int(stats['target_prediction_refreshed'])
        proposed += frame_proposed
        accepted += frame_accepted
    require(len(finished) == 10 and combined == result['combined_frames'] > 0 and
            refreshes == result['prediction_refreshes'] > 0, 'Combined generation and prediction refresh exercised')
    for identity, plan in by_id.items():
        record = records[plan['id']]
        require(record['prompt'] == plan['prompt'] and record['coding'] is plan['coding'], 'Saved full task prompt')
        require(record['generated_ids'] == record['expected_ids'] == generated[identity] == plan['expected'] and
                record['full_token_match'] is True, 'Complete saved reference output')
        require(record['transactions'] == request_transactions[identity], 'Saved task transactions match native frames')
    evaluation = read(root/'evaluation.json')
    require(evaluation['passed'] == evaluation['total'] == 8 and
            evaluation['suite_sha256'] == '675ea667c357f361ce39e3863b61193dc6b32f4a229eb7d29a17803b75408786', 'Frozen sandbox coding gate')
    for plan in controls:
        response_path = root/'responses'/plan['id']/'request-0.result.json'
        response = read(response_path)
        require(response['status'] == 'completed' and response['finish_reason'] == 'stop' and
                response['text'] == plan['reference_text'] and response['generated_ids'] == plan['reference_ids'], 'Complete frozen coding text and IDs')
        rows = [r for r in evaluation['results'] if r['task'] == plan['id']]
        require(len(rows) == 1 and rows[0]['status'] == 'passed' and rows[0]['response_sha256'] == digest(response_path), 'Evaluated exact saved response')
    memory = result['prepared_memory']
    require(len(result['deferred_auxiliary_pack_slots']) == 2 and
            all(type(n) is int and n > 0 for n in result['deferred_auxiliary_pack_slots']), 'Deferred native auxiliary packs exercised')
    require(memory['garnet_live_affine_weight_owners'] == 497 and memory['garnet_live_bf16_weight_owners'] == 8, 'One target and auxiliary weight set')
    require(memory['metal_allocated_bytes'] <= memory['metal_recommended_working_set_bytes']-256*1024*1024, 'Unchanged working-set guard')
    inventory = read(root/'engine/result.json.inventory.json')
    pools = [r for r in inventory['references'] if r['role'] == 'prefill_hidden_pingpong']
    require(len(pools) == len({r['buffer_id'] for r in pools}) == 4 and
            sorted(tuple(r['shape']) for r in pools) == [(2, 10240)]*2+[(2, 655360)]*2,
            'Exactly two independent hidden buffers per verifier/prefill profile')
    pool_ids = {r['buffer_id'] for r in pools}
    require(all(r['role'] == 'prefill_hidden_pingpong' for r in inventory['references'] if r['buffer_id'] in pool_ids), 'Hidden pool cannot alias private cache or graph defaults')
    weights = {}
    for ref in inventory['references']:
        if ref['tag'].startswith('auxiliary_') and ref['role'] == 'weight_record':
            weights.setdefault(ref['tag'], {}).setdefault(ref['name'], []).append(ref['buffer_id'])
    require(set(weights) == {'auxiliary_0', 'auxiliary_1'} and len(weights['auxiliary_0']) == 15 and
            weights['auxiliary_0'] == weights['auxiliary_1'], 'Corrector and teacher share all fifteen raw weights')
    closed = [json.loads(line.split(' ', 1)[1]) for line in (root.parent/(root.name+'.log')).read_text().splitlines()
              if line.startswith('scheduled_mtp_closed ')]
    require(len(closed) == 1, 'Exactly one cleanup record')
    for name in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners', 'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        require(closed[0][name] == 0, 'Closed native owner: '+name)
    return dict(status='passed', checks=checks, coding_passes=8, complete_reference_requests=10,
                output_tokens=sum(len(ids) for ids in generated.values()), native_frames=len(result['frames']),
                combined_frames=combined, prediction_refreshes=refreshes, proposed_drafts=proposed,
                accepted_drafts=accepted, prepared_memory=memory, closed_owners=closed[0],
                result_sha256=digest(root/'engine/result.json'), evaluation_sha256=digest(root/'evaluation.json'),
                auditor_sha256=digest(__file__),
                scope='B2 trained native correctness, sharing and allocation snapshots; no HTTP, prefix or performance admission')


if __name__ == '__main__':
    report = audit(*sys.argv[1:4])
    with Path(sys.argv[4]).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
