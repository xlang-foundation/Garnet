"""Replay source-linked requests, gold, JPEG transforms and prompt IDs on CPU."""
from pathlib import Path
import argparse,ast,base64,collections,hashlib,io,json,sys,zipfile
from mmmu_quality_protocol import PROTOCOL,parse_letter
from fixed_context_workload import study_prompt_ids

sha=lambda b:hashlib.sha256(b).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--source-audit',type=Path,required=True);p.add_argument('--tokenizer-observation',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve(strict=True);assets=args.assets.resolve(strict=True);methods=Path(__file__).resolve().parent
    output=root/'quality-corpus-cpu-verified.json';assert not output.exists()
    summary=json.loads((root/'quality-corpus-summary.json').read_bytes());archive=root/'quality-corpus.zip'
    assert file_sha(archive)==summary['archive_sha256'] and archive.stat().st_size==summary['archive_bytes']
    audit=json.loads(args.source_audit.read_bytes());acquired=json.loads((root/'validation-acquisition.json').read_bytes())
    reader=json.loads((root/'duckdb-reader.json').read_bytes());assert file_sha(root/reader['wheel'])==reader['wheel_sha256']
    sys.path.insert(0,str(root/'duckdb-runtime'));import duckdb,_duckdb,jinja2,tokenizers,PIL
    from PIL import Image,ImageOps
    assert file_sha(Path(_duckdb.__file__))==audit['duckdb_extension_sha256']
    observation=json.loads(args.tokenizer_observation.read_bytes())
    for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','preprocessor_config.json','processor_config.json'):
        assert file_sha(assets/n)==observation['files'][n]['sha256']
    tokenizer=tokenizers.Tokenizer.from_file(str(assets/'tokenizer.json'));encode=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
    env=jinja2.Environment()
    def fail(message):raise ValueError(message)
    env.globals['raise_exception']=fail;template=env.from_string((assets/'chat_template.jinja').read_text(encoding='utf8'))
    connection=duckdb.connect(':memory:',config={'threads':1,'memory_limit':'512MB','autoload_known_extensions':'false','autoinstall_known_extensions':'false','temp_directory':str(root/'duckdb-temp')})
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        raw=z.read('manifest.json');manifest=json.loads(raw)
        assert sha(raw)==summary['manifest_sha256']==file_sha(root/'quality-corpus/manifest.json')
        assert len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        assert set(z.namelist())==set(manifest['files'])|{'manifest.json'}
        assert manifest['status']=='adapted_mmmu_quality_corpus_cpu_prepared_unrun' and manifest['protocol']==PROTOCOL
        assert manifest['revision']==audit['revision']==acquired['revision']=='876ce5cb130f7f7e290ce4d9984357737d4db5cf'
        assert manifest['checkpoint']==observation['checkpoint'] and manifest['checkpoint_revision']==observation['revision']
        assert manifest['source_audit_sha256']==file_sha(args.source_audit)
        assert manifest['acquisition_sha256']==file_sha(root/'validation-acquisition.json')==audit['acquisition_sha256']
        assert manifest['tokenizer_observation_sha256']==file_sha(args.tokenizer_observation) and manifest['tokenizer_sources']==observation['files']
        for n,h in manifest['methods_sha256'].items():assert file_sha(methods/n)==h
        for n,identity in manifest['files'].items():
            data=z.read(n);assert sha(data)==identity['sha256'] and len(data)==identity['bytes']
            assert (root/'quality-corpus'/n).resolve().is_relative_to(root/'quality-corpus')
            assert file_sha(root/'quality-corpus'/n)==identity['sha256']
        assert not manifest['gpu_executed'] and manifest['model_responses']==0 and not manifest['broad_quality_qualified'] and not manifest['performance_admitted']
        assert not manifest['runtime_prompt_ids_verified'] and not manifest['runtime_image_preprocessing_verified']
        gold=json.loads(z.read('gold.json'));expected_gold=[];token_counts=[];subjects=collections.Counter();reasons=collections.Counter();checked=[]
        rows={row['id']:row for row in manifest['rows']};source_rows={row['id']:row for row in audit['records']}
        assert len(rows)==len(source_rows)==900 and set(rows)==set(source_rows)
        try:
            for source in acquired['files']:
                path=root/source['file'];assert file_sha(path)==source['sha256']
                columns=','.join('image_'+str(i)+'."bytes"' for i in range(1,8))
                data=connection.execute('select id,question,options,answer,question_type,'+columns+' from read_parquet(?)',[str(path)]).fetchall()
                for identity,question,options,answer,kind,*images in data:
                    row=rows[identity];old=source_rows[identity];assert row['subject']==old['subject']==source['subject']
                    assert row['source_candidate']==old['candidate_single_image_mc']
                    assert (row['source_question_sha256'],row['source_options_sha256'],row['source_answer_sha256'])==(sha(question.encode()),sha(options.encode()),sha(str(answer).encode()))
                    assert row['excluded_reasons'][:len(old['source_exclusion_reasons'])]==old['source_exclusion_reasons']
                    reasons.update(row['excluded_reasons']);subjects[source['subject']]+=row['request_prepared']
                    if not row['request_prepared']:
                        # The actual frozen corpus has no additional image, prompt or
                        # context exclusions. A future protocol needs a new audit.
                        assert row['excluded_reasons']==old['source_exclusion_reasons'] and not row['source_candidate'];continue
                    assert not row['excluded_reasons'] and row['source_candidate'] and kind=='multiple-choice'
                    choices=ast.literal_eval(options);count=len(choices);present=[i+1 for i,value in enumerate(images) if value]
                    assert present==old['embedded_image_columns']==[row['image_column']] and count==row['option_count']
                    source_image=images[present[0]-1];geometry=row['image_geometry'];assert sha(source_image)==geometry['source_bytes_sha256']
                    with Image.open(io.BytesIO(source_image)) as opened:
                        assert getattr(opened,'n_frames',1)==1 and list(opened.size)==geometry['original_size'] and opened.mode==geometry['original_mode']
                        oriented=ImageOps.exif_transpose(opened).convert('RGBA');assert list(oriented.size)==geometry['oriented_size']
                        rgba=Image.alpha_composite(Image.new('RGBA',oriented.size,'white'),oriented).convert('RGB')
                    # Independent integer half-up geometry, rather than the producer's float scale.
                    longest=max(rgba.size);size=[max(1,min(256,(2*d*256+longest)//(2*longest))) for d in rgba.size]
                    assert size==geometry['resized_size'];offset=[(256-size[0])//2,(256-size[1])//2];assert offset==geometry['canvas_offset']
                    canvas=Image.new('RGB',(256,256),'white');canvas.paste(rgba.resize(tuple(size),Image.Resampling.LANCZOS),tuple(offset))
                    encoded=io.BytesIO();canvas.save(encoded,format='JPEG',quality=100,subsampling=0,optimize=False,progressive=False)
                    jpeg=z.read(row['image_file']);assert encoded.getvalue()==jpeg and sha(jpeg)==geometry['jpeg_sha256'] and len(jpeg)==geometry['jpeg_bytes']
                    with Image.open(io.BytesIO(jpeg)) as decoded:
                        assert decoded.size==(256,256) and sha(decoded.convert('RGB').tobytes())==geometry['pillow_decoded_rgb_sha256']
                    wire=z.read(row['request_file']);body=json.loads(wire);assert sha(wire)==row['canonical_request_sha256']
                    assert set(body)=={'model','messages','max_tokens','temperature','stream','chat_template_kwargs'}
                    assert body['model']=='qwen38-27b-nvfp4' and body['max_tokens']==32 and body['temperature']==0 and body['stream'] is False and body['chat_template_kwargs']=={'enable_thinking':False}
                    content=body['messages'][0]['content'];assert len(body['messages'])==1 and body['messages'][0]['role']=='user'
                    assert len(content)==2 and content[0]['type']=='image_url' and content[1]['type']=='text'
                    assert base64.b64decode(content[0]['image_url']['url'].removeprefix('data:image/jpeg;base64,'),validate=True)==jpeg
                    # Build the expected string without invoking the producer's prompt helper.
                    marker='<image '+str(present[0])+'>'
                    import re
                    def text(value):return re.sub(r'<image\s+'+str(present[0])+r'>','[Image]',value)
                    expected='Question:\n'+text(question)+'\nOptions:\n'+''.join(chr(65+i)+'. '+text(c)+'\n' for i,c in enumerate(choices))+'Answer with exactly one uppercase option letter and no other text.'
                    assert content[1]['text']==expected
                    ids=json.loads(z.read(row['prompt_ids_file']))
                    rendered=template.render(messages=body['messages'],tools=None,add_generation_prompt=True,enable_thinking=False,add_vision_id=False)
                    jinja=[word for value in encode(rendered) for word in ([248056]*64 if value==248056 else [value])]
                    assert ids==jinja==study_prompt_ids(body,encode,248056) and ids.count(248056)==64
                    assert all(type(value) is int and 0<=value<248320 for value in ids)
                    assert row['independent_jinja_exact'] and len(ids)==row['prompt_tokens'] and row['reserved_output_tokens']==32 and len(ids)+32<=4096
                    assert parse_letter(answer,count)==answer
                    expected_gold.append({'id':identity,'answer':answer,'option_count':count,'canonical_request_sha256':sha(wire)});token_counts.append(len(ids));checked.append(identity)
        finally:connection.close()
        assert gold==expected_gold and len(gold)==manifest['requests']==summary['requests']==805
        assert manifest['source_rows']==900 and manifest['excluded_rows']==95 and manifest['structural_candidates']==805
        assert dict(subjects)==manifest['requests_per_subject'] and len(subjects)==30 and dict(reasons)==manifest['exclusion_reason_occurrences']
        assert checked==[r['id'] for r in manifest['rows'] if r['request_prepared']]
    result={'status':'adapted_mmmu_quality_corpus_cpu_replayed','archive_sha256':file_sha(archive),'manifest_sha256':summary['manifest_sha256'],
        'requests_verified':len(checked),'source_rows_verified':900,'source_excluded_rows':95,'subjects':30,
        'prompt_tokens_min':min(token_counts),'prompt_tokens_max':max(token_counts),'reserved_output_tokens':32,'context_capacity':4096,
        'all_jpeg_transforms_regenerated_exact':True,'all_prompt_ids_jinja_native_exact':True,'all_gold_source_labels_exact':True,
        'verifier_sha256':file_sha(Path(__file__)),'gpu_executed':False,'model_responses':0,'broad_quality_qualified':False,'performance_admitted':False,
        'scope':PROTOCOL['scope']}
    output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
