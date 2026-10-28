"""Bounded research serving validation and homogeneous MTP subgroups."""
import re

def validate_body(body,model='qwen38-27b-nvfp4'):
    if not isinstance(body,dict) or body.get('model')!=model or body.get('temperature')!=0:
        raise ValueError('Study requires the pinned model and greedy generation')
    if body.get('chat_template_kwargs')!={'enable_thinking':False}:raise ValueError('Thinking must be disabled')
    limit=body.get('max_tokens')
    if type(limit) is not int or not 2<=limit<=1024:raise ValueError('Output limit outside study profile')
    if type(body.get('ignore_eos',False)) is not bool:raise ValueError('ignore_eos requires an explicit boolean')
    messages=body.get('messages')
    if not isinstance(messages,list) or len(messages)!=1 or messages[0].get('role')!='user':raise ValueError('One user message required')
    content=messages[0].get('content')
    if isinstance(content,str):return
    if not isinstance(content,list) or len(content)!=2 or content[0].get('type')!='image_url' or content[1].get('type')!='text':
        raise ValueError('Image profile requires JPEG first, then text')
    if not isinstance(content[1].get('text'),str) or not content[0].get('image_url',{}).get('url','').startswith('data:image/jpeg;base64,'):
        raise ValueError('Inline JPEG required')

def partition_jobs(jobs,model='qwen38-27b-nvfp4'):
    if type(jobs) is not list or not 1<=len(jobs)<=4:
        raise ValueError('Expected one to four real requests')
    keys=[];groups={}
    for job in jobs:
        if type(job) is not dict or not isinstance(job.get('id'),str) or not re.fullmatch('[0-9a-f]{32}',job['id']):
            raise ValueError('Expected internal request IDs')
        if job['id'] in keys:raise ValueError('Duplicate real request ID')
        keys.append(job['id']);validate_body(job['body'],model)
        key=(job['body']['max_tokens'],job['body'].get('ignore_eos',False))
        groups.setdefault(key,[]).append(job)
    return list(groups.values())
