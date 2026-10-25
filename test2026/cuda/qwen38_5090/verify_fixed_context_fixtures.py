"""Replay prepared corpus with pinned tokenizer/Jinja/JPEG pixels; no inference."""
import argparse,base64,hashlib,json,sys,zipfile
from io import BytesIO
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--source-observation',type=Path,required=True)
    p.add_argument('--preparation-summary',type=Path,required=True);p.add_argument('--short-oracle-archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import jinja2,tokenizers
    from PIL import Image
    sha=lambda b:hashlib.sha256(b).hexdigest();summary=json.loads(args.preparation_summary.read_bytes());observed=json.loads(args.source_observation.read_bytes())
    assert sha(args.archive.read_bytes())==summary['archive_sha256'] and args.archive.stat().st_size==summary['archive_bytes']
    assert summary['status']=='fixed_context_corpus_prepared_unrun'
    for name,info in observed['files'].items():
        if name=='config.json':continue
        raw=(args.assets/name).read_bytes();assert len(raw)==info['bytes'] and sha(raw)==info['sha256']
    tokenizer=tokenizers.Tokenizer.from_file(str(args.assets/'tokenizer.json'));image_id=tokenizer.token_to_id('<|image_pad|>');assert type(image_id) is int
    environment=jinja2.Environment()
    def fail(message):raise ValueError(message)
    environment.globals['raise_exception']=fail;template=environment.from_string((args.assets/'chat_template.jinja').read_text(encoding='utf8'))
    def ids(body):
        rendered=template.render(messages=body['messages'],tools=None,add_generation_prompt=True,enable_thinking=False,add_vision_id=False)
        tokens=tokenizer.encode(rendered,add_special_tokens=False).ids
        return [word for value in tokens for word in ([image_id]*64 if value==image_id else [value])]
    with zipfile.ZipFile(args.archive) as z:
        raw=z.read('manifest.json');manifest=json.loads(raw);assert sha(raw)==summary['manifest_sha256'] and z.testzip() is None
        assert manifest['status']==summary['status'] and manifest['source_observation_sha256']==sha(args.source_observation.read_bytes()) and manifest['tokenizer_sources']==observed['files']
        methods=Path(__file__).resolve().parent
        for name,h in manifest['methods_sha256'].items():assert sha((methods/name).read_bytes())==h
        assert manifest['checkpoint']==observed['checkpoint'] and manifest['revision']==observed['revision']
        assert manifest['short_control_archive_sha256']==sha(args.short_oracle_archive.read_bytes())
        with zipfile.ZipFile(args.short_oracle_archive) as old:
            for control in manifest['short_controls']:
                prefix='results/native-mtp3-http/s3-'+control['case']+'-c1-r0-greedy';req=old.read(prefix+'-request.json');response=old.read(prefix+'-response.json')
                assert sha(req)==control['request_sha256'] and sha(response)==control['response_sha256']
                assert ids(json.loads(req))==json.loads(response)['choices'][0]['prompt_token_ids']
        rows=manifest['rows'];assert len(rows)==len({r['key'] for r in rows})==124 and len(manifest['cohorts'])==8
        expected={'manifest.json'};keys={r['key']:r for r in rows};encodings=set();pixels=set();fixtures={r['key']:r for r in manifest['images']};assert len(fixtures)==62
        for row in rows:
            for field in ('request_file','expected_prompt_ids_file'):
                name=row[field];assert Path(name).name==name;expected.add(name)
            wire=z.read(row['request_file']);tokens=z.read(row['expected_prompt_ids_file']);body=json.loads(wire);expected_ids=json.loads(tokens)
            assert sha(wire)==row['request_sha256'] and sha(tokens)==row['expected_prompt_ids_sha256']
            assert ids(body)==expected_ids and len(expected_ids)==row['prompt_tokens']==3968
            assert row['completion_reserved_tokens']==body['max_tokens']==128 and row['total_context_capacity']==4096 and len(expected_ids)+body['max_tokens']==4096
            assert body['temperature']==0 and body['chat_template_kwargs']=={'enable_thinking':False} and body['stream_options']=={'include_usage':True,'continuous_usage_stats':True}
            if row['case']=='image':
                content=body['messages'][0]['content'];assert content[0]['type']=='image_url' and content[1]['type']=='text' and expected_ids.count(image_id)==64
                raw=base64.b64decode(content[0]['image_url']['url'].removeprefix('data:image/jpeg;base64,'),validate=True)
                with Image.open(BytesIO(raw)) as image:
                    assert image.format=='JPEG' and image.size==(224,224);pixel=sha(image.convert('RGB').tobytes())
                encoded=sha(raw);fixture=fixtures[row['key'].removeprefix('ctx4096-')]
                assert encoded==fixture['encoded_sha256'] and pixel==fixture['decoded_rgb_sha256'] and len(raw)==fixture['bytes']
                assert encoded not in encodings and pixel not in pixels;encodings.add(encoded);pixels.add(pixel)
            else:assert row['case']=='text' and isinstance(body['messages'][0]['content'],str) and image_id not in expected_ids
        assert len(z.namelist())==len(set(z.namelist()))==len(expected)==249 and set(z.namelist())==expected
        visited=[]
        for cohort in manifest['cohorts']:
            case=cohort['case'];c=cohort['concurrency'];assert case in ('text','image') and c in (1,2,4,8) and len(cohort['warmups'])==2 and len(cohort['measured'])==8
            for wave,group in enumerate(cohort['warmups']):
                assert len(group)==c
                for key in group:assert keys[key]['phase']=='warmup' and keys[key]['wave']==wave and keys[key]['case']==case and keys[key]['concurrency']==c;visited.append(key)
            for key in cohort['measured']:assert keys[key]['phase']=='measured' and keys[key]['case']==case and keys[key]['concurrency']==c;visited.append(key)
        assert len(visited)==len(set(visited))==124 and set(visited)==set(keys) and len(encodings)==len(pixels)==62
    report={'status':'fixed_context_corpus_cpu_replayed','archive_sha256':sha(args.archive.read_bytes()),'manifest_sha256':summary['manifest_sha256'],
        'requests_verified':124,'planned_measurement_requests':64,'planned_warmup_requests':60,'input_tokens_each':3968,'reserved_output_tokens_each':128,
        'unique_encoded_and_decoded_image_nonces':62,'short_retained_prompt_id_controls':2,'independent_jinja_full_corpus_exact':True,
        'verifier_sha256':sha(Path(__file__).read_bytes()),'gpu_executed':False,'model_qualified':False,'performance_admitted':False,'scope':manifest['scope']}
    assert not args.output.exists();args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
