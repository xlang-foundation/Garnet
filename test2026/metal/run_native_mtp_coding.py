"""Full frozen coding output replay; native draft/verify/correct transactions."""
import json
import os
import sys
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[3])
import garnet
from xModel.qwen38.vl_27b.metal_mtp_study import MetalMTPStudy

with open(sys.argv[1]) as stream:
    f = json.load(stream)
with open(f['config']) as stream:
    config = json.load(stream)
with open(sys.argv[5]) as stream:
    coding = json.load(stream)
study = None
records = []
try:
    study = MetalMTPStudy(f['checkpoint'], config, sys.argv[4], f['capacity'])
    tokenizer = garnet.tokenizer()
    tokenizer.open(os.path.dirname(f['config']))
    stops = [tokenizer.token_id('<|im_end|>'), tokenizer.token_id('<|endoftext|>')]
    inventory = f['output']+'.inventory.json'
    with open(inventory, 'w') as stream:
        stream.write(study.inventory())
    prepared = json.loads(garnet.memory_stats_json())
    assert prepared['metal_allocated_bytes'] <= prepared['metal_recommended_working_set_bytes']-256*1024*1024
    print('native_mtp_coding_prepared', json.dumps(prepared), flush=True)
    os.mkdir(coding['responses'])
    for plan in coding['plans']:
        study.reset()
        ids = plan['prompt_ids']
        limit = plan['max_output']
        assert len(ids)+limit <= f['capacity']
        generated = [study.prefill(ids)]
        prefill = json.loads(study.session.mtp_stats_json())
        assert prefill['committed'] == [len(ids)] and prefill['profile_chunk'] == 128
        transactions = []
        while len(generated) < limit and generated[-1] not in stops:
            remaining = limit-len(generated)
            result = study.session.mtp_generate([generated[-1]], [remaining], stops, None)[0]
            stats = json.loads(study.session.mtp_generation_stats_json())
            assert result[1] == len(result[0]) and 1 <= result[1] <= min(4, remaining)
            assert stats['draft_profile_chunk'] == 1 and stats['correction_profile_chunk'] == 4
            assert result[2] == (result[0][-1] in stops)
            transactions.append(dict(emitted=result[0], eos=result[2], remaining_before=remaining, stats=stats))
            generated += result[0]
        text_ids = generated[:-1] if generated[-1] in stops else generated
        text = tokenizer.decode(text_ids).encode('utf8').decode('utf8', 'replace')
        record = dict(id=plan['id'], prompt_ids=ids, generated_ids=generated,
                      full_token_match=generated == plan['reference_ids'], text_match=text == plan['reference_text'],
                      prefill=prefill, transactions=transactions, memory=json.loads(garnet.memory_stats_json()))
        records.append(record)
        folder = coding['responses']+'/'+plan['id']
        os.mkdir(folder)
        with open(folder+'/request-0.result.json', 'w') as stream:
            json.dump(dict(status='completed', finish_reason='stop' if generated[-1] in stops else 'length', text=text,
                           generated_ids=generated, generated_tokens=len(generated), prompt_tokens=len(ids)), stream, indent=2)
        with open(f['output'], 'w') as stream:
            json.dump(dict(passed=False, pending=len(coding['plans'])-len(records), tasks=records, prepared_memory=prepared, inventory=inventory,
                           scope='B1 frozen8-task coding replay; no batching, prefix or performance qualification'), stream, indent=2)
        print('native_mtp_coding_task', json.dumps(dict(id=plan['id'], tokens=len(generated), transactions=len(transactions),
            accepted_drafts=sum(x['stats']['accepted_drafts'] for x in transactions), full_token_match=record['full_token_match'], text_match=record['text_match'])), flush=True)
        assert record['full_token_match'] and record['text_match'], 'Frozen complete output changed: '+plan['id']
    with open(f['output'], 'w') as stream:
        json.dump(dict(passed=True, tasks=records, prepared_memory=prepared, inventory=inventory,
                       scope='B1 frozen8-task coding/full-ID gate; no batching, prefix or performance qualification'), stream, indent=2)
finally:
    if study is not None:
        study.close()
    garnet.shutdown()
    memory = json.loads(garnet.memory_stats_json())
    for name in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners', 'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        assert memory[name] == 0, (name, memory[name])
    print('native_mtp_coding_closed', json.dumps(memory), flush=True)
