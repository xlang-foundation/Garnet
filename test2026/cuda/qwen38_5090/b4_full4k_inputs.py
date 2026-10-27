"""Select retained real HTTP inputs for future B4 full-context correctness only."""
import hashlib,io,json,zipfile
CORPUS_SHA256='596991dd0907ccb8ae8ffb090737f59ba8da3faac181bb18fadc74ae0e4ef3ce'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def select_groups(manifest,read):
    rows={row['key']:row for row in manifest['rows']};assert len(rows)==len(manifest['rows'])
    collected={};groups=[]
    for case in ('text','image'):
        cohorts=[c for c in manifest['cohorts'] if c['case']==case and c['concurrency']==8];assert len(cohorts)==1
        keys=cohorts[0]['measured'];assert len(keys)==len(set(keys))==8
        chosen=keys[:4]
        for key in chosen:
            row=rows[key];assert row['case']==case and row['phase']=='measured' and row['concurrency']==8
            assert row['independent_jinja_exact'] is True and row['ignore_eos'] is True
            assert (row['prompt_tokens'],row['completion_reserved_tokens'],row['total_context_capacity'])==(3968,128,4096)
            body_raw=read(row['request_file']);ids_raw=read(row['expected_prompt_ids_file'])
            assert sha(body_raw)==row['request_sha256'] and sha(ids_raw)==row['expected_prompt_ids_sha256']
            body=json.loads(body_raw);ids=json.loads(ids_raw)
            assert len(ids)==3968 and all(type(n) is int and 0<=n<248320 for n in ids)
            assert body['ignore_eos'] is True and type(body['max_tokens']) is int and body['max_tokens']==128 and body['temperature']==0 and body['chat_template_kwargs']['enable_thinking'] is False
            assert len(body['messages'])==1 and body['messages'][0]['role']=='user'
            content=body['messages'][0]['content']
            if case=='text':assert isinstance(content,str)
            else:
                assert isinstance(content,list) and len(content)==2 and content[0]['type']=='image_url' and content[1]['type']=='text'
                assert isinstance(content[1]['text'],str) and content[0]['image_url']['url'].startswith('data:image/')
            assert key not in collected
            collected[key]=dict(key=key,case=case,prompt_ids=ids,body=body,request_sha256=sha(body_raw),expected_prompt_ids_sha256=sha(ids_raw))
        groups.append(dict(label='full4k-'+case,keys=list(chosen),generated_IDs=128,total_context_capacity=4096))
    a,b=groups;groups.append(dict(label='full4k-mixed',keys=[a['keys'][0],b['keys'][0],a['keys'][1],b['keys'][1]],generated_IDs=128,total_context_capacity=4096))
    assert len(collected)==8 and all(len(g['keys'])==len(set(g['keys']))==4 for g in groups)
    return dict(status='retained_fixed4k_B4_correctness_inputs_selected_not_executed',inputs=collected,groups=groups,distinct_B1_oracles_required=8,new_prompts_created=False,benchmark_executed=False)

def load_corpus(raw):
    assert sha(raw)==CORPUS_SHA256
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        return select_groups(json.loads(z.read('manifest.json')),z.read)
