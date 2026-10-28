"""Optional Python call boundaries for diagnostics; preserves model arguments.

Intervals are host wall time, including instrumentation. Graph calls can enqueue
asynchronous GPU work; these intervals must never be labeled CUDA kernel time.
Only the explicitly supplied session is modified. No factory installs this.
"""

class PhaseCall:
    def __init__(self,call,recorder):self.call=call;self.recorder=recorder
    def __call__(self,*args,**kwargs):
        if self.recorder.active is None:return self.call(*args,**kwargs)
        phase=kwargs.get('stage',args[3] if len(args)>3 else None)
        label='mtp_'+phase if phase in ('prefill','draft','correction') else 'mtp_other'
        span=self.recorder.begin(label);ok=False
        try:
            value=self.call(*args,**kwargs);ok=True;return value
        finally:self.recorder.end(span,ok)


class StageAttachment:
    def __init__(self,session,recorder):
        if recorder.active is not None:raise ValueError('Attach before requests start')
        self.recorder=recorder;self.saved=[];self.closed=False
        decoder=session.decoder;verifier=session.verifier
        calls=[(decoder,'chunk_hidden','target_chunk'),(session,'prefill','session_prefill'),
               (session,'generate','session_generate'),(session,'reset','reset'),
               (session,'_draft','draft_proposals'),(session,'_correct','draft_correction'),
               (verifier,'verify','target_verify'),(verifier,'commit_prefix','target_commit'),
               (verifier,'_read','target_norm_head'),(session,'_call',None)]
        calls += [(program,'run','target_program_t'+str(size)) for size,program in sorted(decoder.programs.items())]
        calls += [(session.program,'run','draft_decode_program'),(session.prefill16_program,'run','draft_prefill_program')]
        planned=[]
        for obj,name,label in calls:
            original=getattr(obj,name);state=obj.__dict__
            if not callable(original):raise ValueError('Expected Python call boundary '+name)
            wrapper=recorder.wrap(original,label) if label is not None else PhaseCall(original,recorder)
            planned.append((obj,name,name in state,state.get(name),wrapper))
        try:
            for obj,name,present,before,wrapper in planned:
                setattr(obj,name,wrapper);self.saved.append((obj,name,present,before,wrapper))
        except BaseException:self.close();raise

    def close(self):
        if self.closed:return
        if self.recorder.active is not None:raise ValueError('Finish the request before detaching')
        changed=[]
        for obj,name,present,before,wrapper in reversed(self.saved):
            if obj.__dict__.get(name) is not wrapper:
                changed.append(name);continue
            if present:setattr(obj,name,before)
            else:delattr(obj,name)
        self.closed=True
        if changed:raise ValueError('Preserved independently changed call boundaries: '+','.join(changed))
