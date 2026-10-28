"""CPU source coverage audit only; no inference or quality-promotion threshold."""
from pathlib import Path
import argparse,ast,collections,hashlib,json,re,sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve(strict=True);acquired=json.loads((root/'validation-acquisition.json').read_bytes())
    assert acquired['status']=='mmmu_validation_sources_acquired' and len(acquired['files'])==30 and acquired['revision']=='876ce5cb130f7f7e290ce4d9984357737d4db5cf'
    reader=json.loads((root/'duckdb-reader.json').read_bytes());wheel=root/reader['wheel'];assert sha(wheel)==reader['wheel_sha256']
    sys.path.insert(0,str(root/'duckdb-runtime'));import duckdb,_duckdb
    assert duckdb.__version__==reader['version']
    connection=duckdb.connect(':memory:',config={'threads':1,'memory_limit':'512MB','autoload_known_extensions':'false','autoinstall_known_extensions':'false','temp_directory':str(root/'duckdb-temp')})
    columns=','.join('octet_length(image_'+str(i)+'."bytes")' for i in range(1,8))
    paths=','.join('image_'+str(i)+'."path" is not null' for i in range(1,8))
    records=[];subjects={};types=collections.Counter();images=collections.Counter();exclusions=collections.Counter();ids=set()
    try:
        for f in acquired['files']:
            path=root/f['file'];assert path.resolve().is_relative_to(root/'parquet') and sha(path)==f['sha256'] and path.stat().st_size==f['bytes']
            data=connection.execute('select id,question,options,answer,question_type,subfield,'+columns+','+paths+' from read_parquet(?)',[str(path)]).fetchall()
            assert len(data)==30
            eligible=0
            for row in data:
                identity,question,options,answer,kind,subfield=row[:6];sizes=row[6:13];external=row[13:20]
                assert identity.startswith('validation_'+f['subject']+'_') and identity not in ids;ids.add(identity)
                present=[i+1 for i,size in enumerate(sizes) if size is not None and size>0]
                types[kind]+=1;images[len(present)]+=1;reasons=[]
                if kind!='multiple-choice':reasons.append('question_type_not_multiple_choice')
                if len(present)!=1:reasons.append('embedded_image_count_not_one')
                if any(flag and (size is None or size==0) for flag,size in zip(external,sizes)):reasons.append('external_image_path_without_bytes')
                choices=None
                try:
                    choices=ast.literal_eval(options)
                    assert isinstance(choices,list) and 2<=len(choices)<=26 and all(isinstance(value,str) for value in choices)
                except (ValueError,SyntaxError,TypeError,AssertionError):reasons.append('invalid_or_unsupported_options')
                if kind=='multiple-choice' and (not isinstance(answer,str) or len(answer)!=1 or not choices or answer not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(choices)]):reasons.append('invalid_multiple_choice_gold_label')
                references={int(n) for n in re.findall(r'<image\s+(\d+)>',question+' '+options)}
                if not references.issubset(set(present)):reasons.append('unresolved_image_reference')
                candidate=not reasons;eligible+=candidate;exclusions.update(reasons)
                records.append({'id':identity,'subject':f['subject'],'question_type':kind,'subfield':subfield,
                    'embedded_image_columns':present,'image_bytes':list(sizes),'option_count':len(choices) if isinstance(choices,list) else None,
                    'question_sha256':hashlib.sha256(question.encode()).hexdigest(),'options_sha256':hashlib.sha256(options.encode()).hexdigest(),
                    'answer_sha256':hashlib.sha256(str(answer).encode()).hexdigest(),'candidate_single_image_mc':candidate,'source_exclusion_reasons':reasons})
            subjects[f['subject']]={'rows':len(data),'candidate_single_image_mc':eligible}
    finally:connection.close()
    assert len(records)==len(ids)==900
    candidate_count=sum(r['candidate_single_image_mc'] for r in records)
    report={'status':'mmmu_validation_cpu_source_coverage_audited','dataset':'MMMU/MMMU','revision':acquired['revision'],
        'primary_source':'https://huggingface.co/datasets/MMMU/MMMU/tree/'+acquired['revision'],
        'reported_dataset_license':'apache-2.0','method_sha256':sha(Path(__file__)),'acquisition_sha256':sha(root/'validation-acquisition.json'),
        'duckdb_version':duckdb.__version__,'duckdb_extension_sha256':sha(Path(_duckdb.__file__)),'duckdb_wheel_sha256':reader['wheel_sha256'],
        'source_files':[{k:f[k] for k in ('subject','source_path','bytes','sha256','url')} for f in acquired['files']],
        'rows':len(records),'subjects':subjects,'question_types':dict(types),'embedded_image_count_distribution':dict(images),
        'candidate_single_image_mc':candidate_count,'excluded_source_rows':900-candidate_count,
        'exclusion_reason_occurrences':dict(exclusions),'records':records,'gpu_work':False,'inference_requests':0,
        'broad_quality_qualified':False,'performance_admitted':False,
        'scope':'Source-only structural coverage audit. Candidate single-embedded-image multiple-choice subset is chosen before any model response; actual prompt IDs/context, image preprocessing and request/scoring protocol remain unprepared. Not the full official900-question score or leaderboard protocol. Original Parquets/images/questions remain in one private local cache, not in Git. Any future subset/resizing/prompt/scoring adaptations and exclusions must be explicit; no arbitrary accuracy waiver or model-quality claim.'}
    assert not a.output.exists();a.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:report[k] for k in ('status','rows','question_types','embedded_image_count_distribution','candidate_single_image_mc','excluded_source_rows','exclusion_reason_occurrences','gpu_work')},indent=2))


if __name__=='__main__':main()
