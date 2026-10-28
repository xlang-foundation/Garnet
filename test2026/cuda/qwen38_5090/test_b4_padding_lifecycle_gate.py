import base64,copy,hashlib,json,unittest
from . import test_b4_mtp256_full4k_gate as old
from . import b4_full4k_exact_gate as original_gate
from .b4_padding_lifecycle_inputs import select_lifecycle
from .b4_padding_lifecycle_mtp_gate import admit
from .b4_mtp_public_stream import PublicBatchStream
from .native_worker_io import emit
sha=lambda b:hashlib.sha256(b).hexdigest()

def fixture():
    source,reference,candidate,files=old.MTPFullContext().fixture();files=dict(files)
    inputs=copy.deepcopy(source['inputs']);image=b'\xff\xd8\xff'
    for i,(key,value) in enumerate(inputs.items()):
        content='text' if i<4 else [dict(image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(image).decode())),dict(text='image')]
        value['body']=dict(ignore_eos=True,max_tokens=128,messages=[dict(content=content)])
    selection=select_lifecycle(inputs);reference['inputs']=copy.deepcopy(inputs);candidate['inputs']=copy.deepcopy(inputs)
    candidate['status']='B4_padding_lifecycle_raw_capture_needs_CPython_admission';candidate['default_stop_ids']=[99,100]
    metrics=copy.deepcopy(candidate['rows'][0]['speculative_metrics']);candidate['rows']=[];oracle={r['keys'][0]:r for r in reference['rows']}
    for index,group in enumerate(selection['groups']):
        refs=[oracle[k] for k in group['keys']];row=dict(label=group['label'],keys=group['keys'],real_keys=group['real_keys'],real_rows=group['real_rows'],public_ids=group['public_ids'],generated_ids=[list(r['generated_ids'][0]) for r in refs],speculative_metrics=copy.deepcopy(metrics))
        for phase in ('reset_state','prefill_state','final_state','final_reset'):
            capture=copy.deepcopy(refs[0][phase]);logic=capture['logical']
            for field in ('slots','physical_row_sha256','logical_row_sha256'):logic[field]=[r[phase]['logical'][field][0] for r in refs]
            for i,component in enumerate(logic['components']):component['row_sha256']=[r[phase]['logical']['components'][i]['row_sha256'][0] for r in refs]
            for kind in capture['vectors']:
                raw=b''.join(files[r[phase]['vectors'][kind]['file']] for r in refs);name=group['label']+'-'+phase+'-'+kind+'.f32';files[name]=raw;capture['vectors'][kind]=dict(file=name,bytes=len(raw),sha256=sha(raw))
            row[phase]=capture
        wire=bytearray()
        def sink(event):
            def write(block):wire.extend(block);return len(block)
            emit(event,write)
        n=group['real_rows'];jobs=[dict(id=pid,body=inputs[k]['body']) for pid,k in zip(group['public_ids'],group['real_keys'])];images=[None if isinstance(inputs[k]['body']['messages'][0]['content'],str) else sha(image) for k in group['real_keys']]
        ticks=iter(range(100000));public=PublicBatchStream(jobs,[inputs[k]['prompt_ids'] for k in group['real_keys']],[99,100],lambda ids:','.join(map(str,ids)),sink,images,clock=lambda:next(ticks))
        for at in range(128):
            for physical in range(4):
                public.notify(physical,row['generated_ids'][physical][at],at+1)
                if at==0:public.notify(physical,row['generated_ids'][physical][at],1)
        public.complete(dict(generated_token_ids=row['generated_ids'],stop_token_ids=[None]*4,sampled_tokens_including_stop=[128]*4,speculative_metrics=row['speculative_metrics']))
        name='case'+str(index)+'-public.jsonl';files[name]=bytes(wire);row.update(public_event_file=name,public_event_sha256=sha(bytes(wire)));candidate['rows'].append(row)
    return selection,reference,candidate,files

class LifecycleGateControls(unittest.TestCase):
    def run_gate(self,s,r,c,f):return admit(s,r,c,lambda key,name:f[name],lambda label,name:f[name])
    def test_all_physical_rows_and_real_public_events_exact_without_waiving_original_gate(self):
        s,r,c,f=fixture();q=self.run_gate(s,r,c,f);self.assertEqual(q['target']['full_phase_row_comparisons'],80);self.assertEqual(q['target']['physical_ID_values'],2560);self.assertEqual(q['target']['public_ID_values'],1664)
        self.assertEqual([v['real_rows'] for v in q['target']['public_streams']],[1,2,3,4,3]);self.assertFalse(q['target']['HTTP_quality_or_performance_qualified'])
        with self.assertRaises(AssertionError):original_gate.admit(s,r,c,lambda k,n:f[n],lambda k,n:f[n])
    def test_dummy_last_state_component_and_last_ID_are_not_ignored(self):
        for kind in ('state','ID'):
            s,r,c,f=fixture()
            if kind=='state':c['rows'][0]['final_state']['logical']['components'][127]['row_sha256'][3]='f'*64
            else:c['rows'][0]['generated_ids'][3][127]+=1
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_padding_mapping_real_count_type_and_public_ID_order_are_pinned(self):
        for kind in ('padding','bool','keys','public'):
            s,r,c,f=fixture()
            if kind=='padding':s['groups'][0]['keys'][3]='b'
            elif kind=='bool':c['rows'][0]['real_rows']=True
            elif kind=='keys':c['rows'][3]['real_keys'].reverse()
            else:c['rows'][1]['public_ids'].reverse()
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_rehashed_public_wire_cannot_insert_dummy_event_or_change_last_token(self):
        for kind in ('dummy','token'):
            s,r,c,f=fixture();row=c['rows'][0];name=row['public_event_file'];events=[json.loads(v) for v in f[name].splitlines()]
            if kind=='dummy':events[0]['id']='f'*32
            else:events[-2]['token_id']+=1
            raw=b''.join((json.dumps(e)+'\n').encode() for e in events);f[name]=raw;row['public_event_sha256']=sha(raw)
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_reset_MTP_commit_counts_and_actual_builder_budget_remain_required(self):
        for kind in ('reset','counts','workspace'):
            s,r,c,f=fixture()
            if kind=='reset':c['rows'][-1]['final_reset']['logical']['components'][127]['row_sha256'][0]='f'*64
            elif kind=='counts':c['rows'][-1]['speculative_metrics']['iterations'][-1]['processed_target_counts'][3]=2
            else:c['mtp_memory']['prefill16_graph']['builder_workspace_limit_bytes']=128<<20
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)

if __name__=='__main__':unittest.main()
