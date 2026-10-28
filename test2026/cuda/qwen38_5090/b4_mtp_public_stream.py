"""Public JSON events for a bounded B4 MTP study; dummy rows emit nothing."""
import re,time
from .fixed_length_policy import generation_stop_ids
from .padded_mtp_policy import use_padded_mtp,is_public_mtp_row


class PublicBatchStream:
    def __init__(self,jobs,prompt_ids,stop_ids,decode,emit,image_hashes=None,started=None,clock=time.monotonic):
        if type(jobs) is not list or not 1<=len(jobs)<=4 or len(prompt_ids)!=len(jobs):
            raise ValueError('Expected one to four real requests and prompt-ID lists')
        keys=[job['id'] for job in jobs]
        if any(not isinstance(key,str) or not re.fullmatch('[0-9a-f]{32}',key) for key in keys) or len(set(keys))!=len(keys):
            raise ValueError('Unique real request IDs required')
        limits=[job['body']['max_tokens'] for job in jobs]
        if not use_padded_mtp(limits):raise ValueError('MTP stream requires equal real output limits')
        stop_sets=[generation_stop_ids(job['body'],stop_ids) for job in jobs]
        if any(value!=stop_sets[0] for value in stop_sets):raise ValueError('MTP stream requires equal stop policies')
        for ids,limit in zip(prompt_ids,limits):
            if type(ids) is not list or not ids or len(ids)+limit>4096 or any(type(t) is not int or not 0<=t<248320 for t in ids):
                raise ValueError('Prompt IDs outside the pinned context or vocabulary')
        self.jobs=[dict(id=key,body=dict(max_tokens=limit,ignore_eos=job['body'].get('ignore_eos',False))) for key,limit,job in zip(keys,limits,jobs)]
        self.prompt_ids=[list(ids) for ids in prompt_ids];self.stop_ids=stop_sets[0]
        self.default_stops=set(stop_ids);self.decode=decode;self.emit=emit;self.clock=clock
        self.started=clock() if started is None else started;self.limit=limits[0]
        self.generated=[[] for _ in jobs];self.text=['' for _ in jobs];self.finished=False
        self.images=[None]*len(jobs) if image_hashes is None else list(image_hashes)
        if len(self.images)!=len(jobs) or any(h is not None and (not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h)) for h in self.images):
            raise ValueError('One verified JPEG hash per real row required')

    def notify(self,row,token,sampled):
        if self.finished:raise ValueError('Completed stream cannot emit tokens')
        if not is_public_mtp_row(row,len(self.jobs)):return
        if type(token) is not int or not 0<=token<248320 or type(sampled) is not int or not 1<=sampled<=self.limit:
            raise ValueError('Invalid native token notification')
        ids=self.generated[row]
        if sampled==1 and len(ids)==1:
            if ids[0]!=token:raise ValueError('First prefill token disagrees with generation')
            return
        if sampled!=len(ids)+1:raise ValueError('Native token notifications are not contiguous')
        ids.append(token);self.text[row]=self.decode([t for t in ids if t not in self.default_stops])
        self.emit(dict(event='token',id=self.jobs[row]['id'],token_id=token,sampled_tokens=sampled,
                       prompt_tokens=len(self.prompt_ids[row]),text=self.text[row],stop=token in self.stop_ids,
                       native_monotonic_seconds=self.clock()))

    def complete(self,result):
        if self.finished:raise ValueError('Stream already completed')
        for key in ('generated_token_ids','stop_token_ids','sampled_tokens_including_stop'):
            if len(result[key])!=4:raise ValueError('MTP result requires four physical rows')
        for row in range(len(self.jobs)):
            expected=list(result['generated_token_ids'][row]);stop=result['stop_token_ids'][row]
            if any(type(t) is not int or not 0<=t<248320 for t in expected) or type(result['sampled_tokens_including_stop'][row]) is not int:
                raise ValueError('Invalid native result token/count types')
            if stop is not None:
                if type(stop) is not int or stop not in self.stop_ids:raise ValueError('MTP stopped outside the request stop policy')
                expected.append(stop)
            if not 1<=len(expected)<=self.limit or (stop is None and len(expected)!=self.limit):
                raise ValueError('Native result violates the requested output length')
            if expected!=self.generated[row] or result['sampled_tokens_including_stop'][row]!=len(expected):
                raise ValueError('Native result disagrees with public token events')
        self.finished=True
        for row,job in enumerate(self.jobs):
            self.emit(dict(event='completed',id=job['id'],generated_ids=list(self.generated[row]),
                           prompt_ids=list(self.prompt_ids[row]),text=self.text[row],image_sha256=self.images[row],
                           image_grid=[1,16,16] if self.images[row] else None,generation_mode='verified_mtp',
                           speculative_metrics=result['speculative_metrics'],physical_batch=4,real_rows=len(self.jobs),
                           padded_rows=4-len(self.jobs),padding_policy='duplicate_first_real_row_no_public_dummy_events',
                           ignore_eos=job['body'].get('ignore_eos',False),
                           finish_reason='stop' if result['stop_token_ids'][row] is not None else 'length',
                           native_elapsed_seconds=self.clock()-self.started))
