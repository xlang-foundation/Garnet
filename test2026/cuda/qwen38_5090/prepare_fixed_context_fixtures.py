"""CPU-only exact4K request preparation; no inference, HTTP or GPU initialization."""
import argparse,hashlib,json,platform,sys,types,zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--source-observation',type=Path,required=True)
    parser.add_argument('--short-oracle-archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();methods=Path(__file__).resolve().parent;sys.path.insert(0,str(methods))
    import jinja2,tokenizers,PIL
    from benchmark_stream import prepare_workloads
    from fixed_context_workload import fixed_context_request,study_prompt_ids
    sha=lambda b:hashlib.sha256(b).hexdigest();observation=json.loads(args.source_observation.read_bytes())
    names=('tokenizer.json','tokenizer_config.json','chat_template.jinja','preprocessor_config.json','processor_config.json')
    for name in names:
        raw=(args.assets/name).read_bytes();assert len(raw)==observation['files'][name]['bytes'] and sha(raw)==observation['files'][name]['sha256']
    config_raw=(methods/'evidence/nvfp4-config.json').read_bytes();assert sha(config_raw)==observation['files']['config.json']['sha256'];config=json.loads(config_raw)
    tokenizer=tokenizers.Tokenizer.from_file(str(args.assets/'tokenizer.json'));encode=lambda text:tokenizer.encode(text,add_special_tokens=False).ids
    image_id=config['image_token_id'];assert tokenizer.token_to_id('<|image_pad|>')==image_id
    environment=jinja2.Environment()
    def fail(message):raise ValueError(message)
    environment.globals['raise_exception']=fail;template=environment.from_string((args.assets/'chat_template.jinja').read_text(encoding='utf8'))
    def rendered_ids(body):
        text=template.render(messages=body['messages'],tools=None,add_generation_prompt=True,enable_thinking=False,add_vision_id=False)
        return [word for value in encode(text) for word in ([image_id]*64 if value==image_id else [value])]
    controls=[]
    with zipfile.ZipFile(args.short_oracle_archive) as z:
        for case in ('text','image'):
            base='results/native-mtp3-http/s3-'+case+'-c1-r0-greedy';req=z.read(base+'-request.json');response=z.read(base+'-response.json');body=json.loads(req);ids=json.loads(response)['choices'][0]['prompt_token_ids']
            assert study_prompt_ids(body,encode,image_id)==rendered_ids(body)==ids
            controls.append({'case':case,'request_sha256':sha(req),'response_sha256':sha(response),'tokens':len(ids),'retained_ids_exact':True,'independent_jinja_exact':True})
    settings=dict(cases=['text','image'],concurrency=[1,2,4,8],warmups=2,requests=8,workload_seed=3,model='qwen38-27b-nvfp4',max_tokens=128,image_format='jpeg',image_order='image-first',image_nonce_pattern='binary-grid')
    workloads,images=prepare_workloads(types.SimpleNamespace(**settings));rows=[];cohorts=[];raw_files={}
    for (case,concurrency),group in workloads.items():
        cohort={'case':case,'concurrency':concurrency,'warmups':[],'measured':[]};cohorts.append(cohort)
        for phase,groups in [('warmup',group['warmups']),('measured',[group['measured']])]:
            for wave,entries in enumerate(groups):
                keys=[]
                for key,body in entries:
                    body,ids,info=fixed_context_request(body,encode,image_id);assert len(ids)==3968 and ids==rendered_ids(body)
                    key='ctx4096-'+key;wire=json.dumps(body,separators=(',',':')).encode();tokens=json.dumps(ids,separators=(',',':')).encode()
                    request_name=key+'-request.json';ids_name=key+'-prompt-ids.json';raw_files[request_name]=wire;raw_files[ids_name]=tokens;keys.append(key)
                    rows.append({'key':key,'phase':phase,'wave':wave if phase=='warmup' else None,'case':case,'concurrency':concurrency,
                        'request_file':request_name,'request_sha256':sha(wire),'expected_prompt_ids_file':ids_name,'expected_prompt_ids_sha256':sha(tokens),
                        'independent_jinja_exact':True,**info})
                if phase=='warmup':cohort['warmups'].append(keys)
                else:cohort['measured']=keys
    assert len(rows)==124 and len(images)==62 and len({r['key'] for r in rows})==124
    manifest={'status':'fixed_context_corpus_prepared_unrun','checkpoint':observation['checkpoint'],'revision':observation['revision'],
        'source_observation_sha256':sha(args.source_observation.read_bytes()),'tokenizer_sources':observation['files'],
        'methods_sha256':{name:sha((methods/name).read_bytes()) for name in ['prepare_fixed_context_fixtures.py','fixed_context_workload.py','benchmark_stream.py','probe_reference_http.py']},
        'short_control_archive_sha256':sha(args.short_oracle_archive.read_bytes()),'short_controls':controls,
        'python':platform.python_version(),'tokenizers':tokenizers.__version__,'jinja2':jinja2.__version__,'pillow':PIL.__version__,
        'settings':settings,'cohorts':cohorts,'rows':rows,'images':images,
        'scope':'CPU-prepared artificial repeated-note exact3968-input/128-output-reserve requests including chat template and64image tokens. JPEG/pixel nonces retained from existing preparation. Actual server prompt IDs/image tokenization/cache misses, boundary correctness, ordinary/MTP and model quality must be checked before timing. No GPU use, model execution, generated outputs, 4K benchmark result, broad-quality or 2x admission.'}
    assert not args.output.exists();args.output.mkdir(parents=True)
    for name,raw in raw_files.items():(args.output/name).write_bytes(raw)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    archive=args.output.parent/'fixed-context-4096-corpus.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.iterdir()):z.writestr(path.name,path.read_bytes())
    summary={'status':manifest['status'],'archive_sha256':sha(archive.read_bytes()),'archive_bytes':archive.stat().st_size,'manifest_sha256':sha((args.output/'manifest.json').read_bytes()),
        'requests':len(rows),'measured_requests':sum(r['phase']=='measured' for r in rows),'warmup_requests':sum(r['phase']=='warmup' for r in rows),
        'image_nonces':len(images),'input_tokens_each':3968,'reserved_output_tokens_each':128,'short_controls':controls,'scope':manifest['scope']}
    (args.output.parent/'fixed-context-4096-preparation-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
