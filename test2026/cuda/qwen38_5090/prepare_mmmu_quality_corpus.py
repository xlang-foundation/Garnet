"""Freeze adapted MMMU semantic requests and exclusions on CPU before inference.

One fixed private cache holds original Parquets and derived request/image files.
Only compact source hashes/protocol metadata belong in Git. This is deliberately
not an official full MMMU evaluation or a model-quality admission gate.
"""
from pathlib import Path
import argparse,ast,base64,collections,hashlib,io,json,math,platform,sys,zipfile

from mmmu_quality_protocol import PROTOCOL,choice_prompt
from fixed_context_workload import study_prompt_ids

def digest(raw):return hashlib.sha256(raw).hexdigest()
def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def image_bytes(raw):
    from PIL import Image,ImageOps
    with Image.open(io.BytesIO(raw)) as opened:
        if getattr(opened,'n_frames',1)!=1:raise ValueError('Animated image is outside this screening protocol')
        original_size=list(opened.size);original_mode=opened.mode
        image=ImageOps.exif_transpose(opened).convert('RGBA');oriented=list(image.size)
        white=Image.new('RGBA',image.size,(255,255,255,255));white.alpha_composite(image);rgb=white.convert('RGB')
    scale=min(256/rgb.width,256/rgb.height)
    resized=[max(1,min(256,int(math.floor(d*scale+0.5)))) for d in rgb.size]
    rgb=rgb.resize(tuple(resized),Image.Resampling.LANCZOS);offset=[(256-d)//2 for d in resized]
    canvas=Image.new('RGB',(256,256),(255,255,255));canvas.paste(rgb,tuple(offset))
    stream=io.BytesIO();canvas.save(stream,format='JPEG',quality=100,subsampling=0,optimize=False,progressive=False)
    encoded=stream.getvalue()
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert decoded.size==(256,256);pixel_digest=digest(decoded.convert('RGB').tobytes())
    return encoded,{'source_bytes_sha256':digest(raw),'original_size':original_size,'original_mode':original_mode,
        'oriented_size':oriented,'resized_size':resized,'canvas_offset':offset,'jpeg_bytes':len(encoded),
        'jpeg_sha256':digest(encoded),'pillow_decoded_rgb_sha256':pixel_digest}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--assets',required=True,type=Path)
    parser.add_argument('--source-audit',required=True,type=Path);parser.add_argument('--tokenizer-observation',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve(strict=True);assets=args.assets.resolve(strict=True)
    methods=Path(__file__).resolve().parent
    audit=json.loads(args.source_audit.read_bytes());acquired=json.loads((root/'validation-acquisition.json').read_bytes())
    assert audit['status']=='mmmu_validation_cpu_source_coverage_audited' and audit['rows']==900 and audit['candidate_single_image_mc']==805
    assert audit['revision']==acquired['revision']=='876ce5cb130f7f7e290ce4d9984357737d4db5cf'
    assert audit['acquisition_sha256']==file_digest(root/'validation-acquisition.json')
    source_rows={r['id']:r for r in audit['records']};assert len(source_rows)==900
    reader=json.loads((root/'duckdb-reader.json').read_bytes());assert file_digest(root/reader['wheel'])==reader['wheel_sha256']
    sys.path.insert(0,str(root/'duckdb-runtime'));import duckdb,_duckdb,jinja2,tokenizers,PIL
    assert duckdb.__version__==reader['version'] and file_digest(Path(_duckdb.__file__))==audit['duckdb_extension_sha256']
    observation=json.loads(args.tokenizer_observation.read_bytes())
    assert observation['checkpoint']=='nvidia/Qwen3.8-27B-NVFP4' and observation['revision']=='482ca0f3832238542f8f5295dde86b5f22711d80'
    for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','preprocessor_config.json','processor_config.json'):
        assert file_digest(assets/n)==observation['files'][n]['sha256']
    config_path=methods/'evidence/nvfp4-config.json';assert file_digest(config_path)==observation['files']['config.json']['sha256']
    config=json.loads(config_path.read_bytes());image_id=config['image_token_id'];assert image_id==248056
    tokenizer=tokenizers.Tokenizer.from_file(str(assets/'tokenizer.json'));encode=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
    assert tokenizer.token_to_id('<|image_pad|>')==image_id
    env=jinja2.Environment()
    def fail(message):raise ValueError(message)
    env.globals['raise_exception']=fail;template=env.from_string((assets/'chat_template.jinja').read_text(encoding='utf8'))
    output=root/'quality-corpus';assert not output.exists();output.mkdir()
    for name in ('images','requests','prompt-ids'):(output/name).mkdir()
    rows=[];gold=[];reason_counts=collections.Counter();counts=collections.Counter();files={};seen=set()
    def save(relative,raw):
        path=output/relative;assert path.resolve().is_relative_to(output.resolve()) and not path.exists()
        path.write_bytes(raw);files[relative]={'bytes':len(raw),'sha256':digest(raw)}
    def json_bytes(value):return json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()
    connection=duckdb.connect(':memory:',config={'threads':1,'memory_limit':'512MB','autoload_known_extensions':'false','autoinstall_known_extensions':'false','temp_directory':str(root/'duckdb-temp')})
    try:
        for source in acquired['files']:
            path=root/source['file'];assert path.resolve().is_relative_to(root/'parquet') and file_digest(path)==source['sha256']
            columns=','.join('image_'+str(i)+'."bytes"' for i in range(1,8))
            data=connection.execute('select id,question,options,answer,question_type,'+columns+' from read_parquet(?)',[str(path)]).fetchall()
            assert len(data)==30
            for identity,question,options,answer,kind,*images in data:
                assert identity not in seen and identity in source_rows;seen.add(identity);audited=source_rows[identity]
                assert digest(question.encode())==audited['question_sha256'] and digest(options.encode())==audited['options_sha256'] and digest(str(answer).encode())==audited['answer_sha256']
                row={'id':identity,'subject':source['subject'],'source_question_sha256':audited['question_sha256'],
                     'source_options_sha256':audited['options_sha256'],'source_answer_sha256':audited['answer_sha256'],
                     'source_candidate':audited['candidate_single_image_mc'],'excluded_reasons':list(audited['source_exclusion_reasons'])}
                if row['source_candidate']:
                    choices=ast.literal_eval(options);present=[i+1 for i,value in enumerate(images) if value]
                    assert present==audited['embedded_image_columns'] and len(present)==1 and kind=='multiple-choice'
                    row['option_count']=len(choices);row['image_column']=present[0]
                    try:
                        text=choice_prompt(question,choices,present[0]);jpeg,geometry=image_bytes(images[present[0]-1])
                    except (ValueError,OSError) as error:
                        row['excluded_reasons'].append('pre_inference_image_or_prompt_admission_failure');row['admission_error']=repr(error)
                    else:
                        body={'model':'qwen38-27b-nvfp4','messages':[{'role':'user','content':[
                            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()}},
                            {'type':'text','text':text}]}],'max_tokens':32,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
                        ids=study_prompt_ids(body,encode,image_id)
                        rendered=template.render(messages=body['messages'],tools=None,add_generation_prompt=True,enable_thinking=False,add_vision_id=False)
                        jinja_ids=[word for value in encode(rendered) for word in ([image_id]*64 if value==image_id else [value])]
                        assert ids==jinja_ids and ids.count(image_id)==64
                        row.update(prompt_tokens=len(ids),reserved_output_tokens=32,independent_jinja_exact=True,image_geometry=geometry)
                        if len(ids)+32>4096:row['excluded_reasons'].append('prompt_plus_output_reservation_exceeds_4096')
                        else:
                            wire=json_bytes(body);row.update(request_file='requests/'+identity+'.json',prompt_ids_file='prompt-ids/'+identity+'.json',image_file='images/'+identity+'.jpg',canonical_request_sha256=digest(wire))
                            save(row['request_file'],wire);save(row['prompt_ids_file'],json_bytes(ids));save(row['image_file'],jpeg)
                            gold.append({'id':identity,'answer':answer,'option_count':len(choices),'canonical_request_sha256':digest(wire)})
                row['request_prepared']=not row['excluded_reasons'];reason_counts.update(row['excluded_reasons'])
                counts[source['subject']]+=row['request_prepared'];rows.append(row)
                print('MMMU_CPU_REQUEST_SOURCE',identity,'prepared' if row['request_prepared'] else 'excluded',flush=True)
    finally:connection.close()
    assert len(rows)==len(seen)==900 and len(gold)==sum(r['request_prepared'] for r in rows)
    save('gold.json',json_bytes(gold))
    manifest={'status':'adapted_mmmu_quality_corpus_cpu_prepared_unrun','dataset':'MMMU/MMMU','revision':audit['revision'],
        'checkpoint':observation['checkpoint'],'checkpoint_revision':observation['revision'],'protocol':PROTOCOL,
        'source_audit_sha256':file_digest(args.source_audit),'acquisition_sha256':file_digest(root/'validation-acquisition.json'),
        'tokenizer_observation_sha256':file_digest(args.tokenizer_observation),'tokenizer_sources':observation['files'],
        'methods_sha256':{n:file_digest(methods/n) for n in ('prepare_mmmu_quality_corpus.py','mmmu_quality_protocol.py','fixed_context_workload.py')},
        'environment':{'python':platform.python_version(),'duckdb':duckdb.__version__,'duckdb_extension_sha256':file_digest(Path(_duckdb.__file__)),
          'pillow':PIL.__version__,'tokenizers':tokenizers.__version__,'jinja2':jinja2.__version__},
        'source_rows':900,'structural_candidates':805,'requests':len(gold),'excluded_rows':900-len(gold),'requests_per_subject':dict(counts),
        'exclusion_reason_occurrences':dict(reason_counts),'rows':rows,'files':files,'gpu_executed':False,'model_responses':0,
        'runtime_prompt_ids_verified':False,'runtime_image_preprocessing_verified':False,'broad_quality_qualified':False,'performance_admitted':False,
        'scope':PROTOCOL['scope']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    archive=root/'quality-corpus.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n in list(files)+['manifest.json']:z.write(output/n,n)
    summary={k:manifest[k] for k in ('status','dataset','revision','checkpoint','checkpoint_revision','source_rows','structural_candidates','requests','excluded_rows','requests_per_subject','exclusion_reason_occurrences','gpu_executed','model_responses','runtime_prompt_ids_verified','runtime_image_preprocessing_verified','broad_quality_qualified','performance_admitted','scope')}
    summary.update(archive_bytes=archive.stat().st_size,archive_sha256=file_digest(archive),manifest_sha256=file_digest(output/'manifest.json'),protocol=PROTOCOL,
                   methods_sha256=manifest['methods_sha256'],environment=manifest['environment'])
    (root/'quality-corpus-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
