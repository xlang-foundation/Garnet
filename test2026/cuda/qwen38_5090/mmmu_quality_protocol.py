"""Adapted single-image MC screening protocol; never the official MMMU score."""
import re

PROTOCOL={
    'name':'mmmu_validation_single_image_mc_256_letterbox_strict_letter_v1',
    'image_size':[256,256],'image_grid':[1,16,16],'merged_image_tokens':64,
    'image_method':'EXIF transpose; composite transparency on white; aspect-preserving LANCZOS fit; centered white RGB256 canvas; JPEG quality100/subsampling0',
    'prompt_method':'One image then question and A.-labelled choices; sole-image reference markers become [Image]; exact uppercase letter instruction; no thinking',
    'context_capacity':4096,'reserved_output_tokens':32,'temperature':0,
    'score_method':'Strip surrounding whitespace; accept exactly one valid uppercase option letter. Any other complete response is unparseable and incorrect. Never guess/randomize/drop failures.',
    'engine_request_adapters':'Add only installed-engine token-ID return flags to canonical semantic requests, retaining both wire hashes. Model alias may change only for declared INT4/4B comparisons.',
    'required_runtime_checks':'Actual prompt IDs,64image tokens/grid, prompt+completion<=4096, same checkpoint/semantic payload and complete output records; exact preprocessing equality is not established by CPU preparation.',
    'scope':'Pre-inference adapted low-resolution screening subset. Not full official900-question MMMU, a quality-promotion threshold, high-resolution VLM qualification or performance evidence.'}

def choice_prompt(question,options,image_column):
    if not isinstance(question,str) or not question.strip() or not isinstance(options,list) or not 2<=len(options)<=26 or any(not isinstance(s,str) for s in options):
        raise ValueError('Invalid multiple-choice source')
    if type(image_column) is not int or not 1<=image_column<=7:raise ValueError('Invalid sole image column')
    values=[question]+options
    if any('<|' in s or '\x00' in s for s in values):raise ValueError('Source contains reserved chat controls')
    def replace(value):
        def marker(match):
            if int(match.group(1))!=image_column:raise ValueError('Unresolved image reference')
            return '[Image]'
        return re.sub(r'<image\s+(\d+)>',marker,value)
    return 'Question:\n'+replace(question)+'\nOptions:\n'+'\n'.join(chr(65+i)+'. '+replace(value) for i,value in enumerate(options))+'\nAnswer with exactly one uppercase option letter and no other text.'

def parse_letter(text,option_count):
    if type(option_count) is not int or not 2<=option_count<=26:raise ValueError('Invalid option count')
    if not isinstance(text,str):raise ValueError('Response text is missing')
    value=text.strip()
    return value if len(value)==1 and value in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:option_count] else None

def score_complete(gold,outputs):
    """Outputs must explicitly bind every request ID/hash; exceptions stay failures."""
    if not isinstance(gold,list) or not gold:raise ValueError('Empty gold corpus')
    expected={}
    for row in gold:
        identity=row['id']
        if identity in expected or parse_letter(row['answer'],row['option_count'])!=row['answer']:raise ValueError('Invalid/duplicate gold')
        expected[identity]=row
    observed={}
    for row in outputs:
        identity=row['id']
        if identity not in expected or identity in observed:raise ValueError('Unexpected/duplicate response ID')
        source=expected[identity]
        if row['canonical_request_sha256']!=source['canonical_request_sha256']:raise ValueError('Request identity mismatch')
        if row.get('status')!='completed':raise ValueError('Incomplete/error response; no full-corpus accuracy admitted')
        prediction=parse_letter(row.get('text'),source['option_count'])
        observed[identity]={'id':identity,'prediction':prediction,'correct':prediction==source['answer'],'unparseable':prediction is None}
    if set(observed)!=set(expected):raise ValueError('Missing responses; no full-corpus accuracy admitted')
    records=[observed[identity] for identity in expected];correct=sum(r['correct'] for r in records)
    return {'status':'adapted_strict_mc_complete_scored','count':len(records),'correct':correct,'accuracy':correct/len(records),
            'unparseable_incorrect':sum(r['unparseable'] for r in records),'records':records,
            'official_mmmu_score':False,'quality_promotion_admitted':False,'performance_admitted':False}

def paired_counts(gold,baseline,native):
    left=score_complete(gold,baseline);right=score_complete(gold,native);pairs=list(zip(left['records'],right['records']))
    return {'count':left['count'],'baseline_correct':left['correct'],'native_correct':right['correct'],
            'native_minus_baseline_correct':right['correct']-left['correct'],
            'baseline_only_correct':sum(a['correct'] and not b['correct'] for a,b in pairs),
            'native_only_correct':sum(b['correct'] and not a['correct'] for a,b in pairs),
            'same_parsed_letter':sum(a['prediction'] is not None and a['prediction']==b['prediction'] for a,b in pairs),
            'baseline_unparseable':left['unparseable_incorrect'],'native_unparseable':right['unparseable_incorrect'],
            'quality_promotion_admitted':False,'performance_admitted':False}
