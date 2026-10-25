"""Prepare actual token-budgeted requests before timing; never truncate token IDs."""
import copy

SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
HEADER='Background notes for a context-length test. Answer the question at the end.\n'
NOTE='Water evaporates, rises, cools, and condenses into cloud droplets.\n'


def study_prompt_ids(body,encode,image_token_id):
    """Exact current native single-user/no-thinking template, including image slots."""
    if not isinstance(body,dict) or body.get('chat_template_kwargs')!={'enable_thinking':False}:
        raise ValueError('Expected explicit no-thinking study request')
    messages=body.get('messages')
    if not isinstance(messages,list) or len(messages)!=1 or messages[0].get('role')!='user':
        raise ValueError('Expected one user message')
    content=messages[0].get('content')
    if isinstance(content,str):return list(encode('<|im_start|>user\n'+content+SUFFIX))
    if not isinstance(content,list) or len(content)!=2 or content[0].get('type')!='image_url' or content[1].get('type')!='text' or not isinstance(content[1].get('text'),str):
        raise ValueError('Expected a single image followed by text')
    if type(image_token_id) is not int or image_token_id<0 or not content[0].get('image_url',{}).get('url','').startswith('data:image/jpeg;base64,'):
        raise ValueError('Expected pinned image ID and inline JPEG study fixture')
    return list(encode('<|im_start|>user\n<|vision_start|>'))+[image_token_id]*64+list(encode('<|vision_end|>'+content[1]['text']+SUFFIX))


def fixed_context_request(body,encode,image_token_id,*,capacity=4096,output_tokens=128):
    if type(capacity) is not int or type(output_tokens) is not int or not 32<=capacity<=4096 or not 2<=output_tokens<=1024 or output_tokens>=capacity:
        raise ValueError('Invalid bounded context/output reservation')
    if body.get('max_tokens')!=output_tokens:raise ValueError('Request output limit must match reservation')
    original=study_prompt_ids(body,encode,image_token_id);candidate=copy.deepcopy(body)
    content=candidate['messages'][0]['content'];text=content if isinstance(content,str) else content[1]['text']
    if not text or '<|' in text:raise ValueError('Expected nonempty ordinary benchmark text')
    target=capacity-output_tokens
    def make(lines,tail):
        value=HEADER+NOTE*lines+' context'*tail+'\nQuestion:\n'+text
        if isinstance(content,str):candidate['messages'][0]['content']=value
        else:content[1]['text']=value
        return study_prompt_ids(candidate,encode,image_token_id)
    if len(make(0,0))>target:raise ValueError('Request skeleton exceeds prompt budget')
    # For this fixed text, increasing repeated complete notes and single-token
    # filler is monotonic. Require exact re-encoding rather than assume BPE counts.
    def fit(build):
        low,high=0,target
        while low<high:
            middle=(low+high+1)//2
            if len(build(middle))<=target:low=middle
            else:high=middle-1
        return low
    lines=fit(lambda n:make(n,0));tail=fit(lambda n:make(lines,n));ids=make(lines,tail)
    if len(ids)!=target:raise ValueError('Tokenizer cannot exactly fill this prompt budget; no truncation applied')
    return candidate,ids,{'prompt_tokens':len(ids),'completion_reserved_tokens':output_tokens,'total_context_capacity':capacity,
        'original_prompt_tokens':len(original),'complete_background_notes':lines,'tail_filler_units':tail,
        'scope':'Artificial repeated-note context workload; actual input tokens include template and64image slots when present. Not broad quality, generated output, GPU execution or timing evidence.'}
