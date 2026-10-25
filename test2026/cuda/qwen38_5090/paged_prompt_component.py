"""Independent raw-byte oracle and lossless capture replay for paged prompt IO."""
import hashlib,math,struct

HIDDEN=5120
ROW_BYTES=HIDDEN*4
def sha(raw):return hashlib.sha256(raw).hexdigest()

def case_specs():
    lengths=[191,127,64,255,159,128,96,257];starts=[95,31,0,127,-1,64,32,193]
    specs=[]
    for batch,page in ((1,32),(2,64),(4,128),(8,32),(8,64),(8,128)):
        specs.append({'key':f'b{batch}-p{page}-mixed','batch':batch,'page_size':page,'lengths':lengths[:batch],
                      'visual_starts':starts[:batch],'row_ids':list(range(batch)),'visual_epoch':0})
    specs.append({'key':'b8-p32-new-visuals-rotate3','batch':8,'page_size':32,'lengths':[lengths[i] for i in (3,4,5,6,7,0,1,2)],
                  'visual_starts':[starts[i] for i in (3,4,5,6,7,0,1,2)],'row_ids':[3,4,5,6,7,0,1,2],'visual_epoch':1})
    specs.append({'key':'b8-p128-full3968','batch':8,'page_size':128,'lengths':[3968]*8,
                  'visual_starts':[95,127,191,255,-1,3904,0,3903],'row_ids':list(range(8)),'visual_epoch':0})
    return specs

def token_rows(spec,selected_ids):
    assert len(selected_ids)==16 and all(type(t) is int and 0<=t<248320 for t in selected_ids)
    return [[selected_ids[(row*3+t*5)%16] for t in range(length)] for row,length in zip(spec['row_ids'],spec['lengths'])]

def visual_values(epoch,row):
    # Dyadic values distinguish every row/token/channel and both visual epochs.
    return [epoch*64+(row*64+t)/16+(d+1)/131072 for t in range(64) for d in range(HIDDEN)]

def visual_row_bytes(epoch,row,token):
    return struct.pack('<'+str(HIDDEN)+'f',*(epoch*64+(row*64+token)/16+(d+1)/131072 for d in range(HIDDEN)))

def expected_events(spec):
    count=max(spec['lengths']);events=[]
    for start in range(0,count,32):
        end=min(start+32,count)
        events.extend(('target',t) for t in range(start,end))
        events.extend(('shifted',t+1) for t in range(start,min(end,count-1)))
    return events+[('held_after_iteration',0),('held_after_close',0)]

def embedding_oracle(fixture,read):
    assert fixture['status']=='checkpoint_embedding_rows_prepared_cpu_only' and fixture['shape']==[248320,HIDDEN]
    oracle={}
    for token,item in fixture['rows'].items():
        raw=read(item['bf16']);f32=read(item['f32'])
        assert len(raw)==HIDDEN*2 and len(f32)==ROW_BYTES and sha(raw)==item['bf16_sha256'] and sha(f32)==item['f32_sha256']
        # Format-based integer conversion, independent of the fixture writer's
        # byte-prefix construction and the native CUDA embedding implementation.
        words=struct.unpack('<'+str(HIDDEN)+'H',raw)
        assert f32==struct.pack('<'+str(HIDDEN)+'I',*(word<<16 for word in words))
        assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',f32));oracle[int(token)]=f32
    assert set(fixture['ids'])==set(oracle);return oracle

class RowCapture:
    """Store each actual native row byte string once, preserving every read event."""
    def __init__(self,directory):
        self.directory=directory;directory.mkdir(exist_ok=False);self.rows={}
    def record(self,raw,batch):
        if len(raw)!=batch*ROW_BYTES:raise ValueError('Native paged prompt read has wrong byte size')
        ids=[]
        for row in range(batch):
            value=raw[row*ROW_BYTES:(row+1)*ROW_BYTES];digest=sha(value)
            if digest not in self.rows:
                (self.directory/(digest+'.f32')).write_bytes(value);self.rows[digest]=len(value)
            else:assert self.rows[digest]==len(value) and (self.directory/(digest+'.f32')).read_bytes()==value
            ids.append(digest)
        return ids

def replay_case(record,events,read_blob,fixture,oracle,spec):
    assert record['spec']==spec and type(record['closed']) is bool and record['closed'] is True
    expected=expected_events(spec);assert len(events)==record['read_events']==len(expected)
    selected=token_rows(spec,fixture['ids']);cache={};visuals={};logical_bytes=0
    for event,(kind,t) in zip(events,expected):
        assert event['kind']==kind and type(event['token']) is int and event['token']==t and isinstance(event['rows'],list) and len(event['rows'])==spec['batch']
        for b,digest in enumerate(event['rows']):
            assert isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
            if digest not in cache:
                raw=read_blob(digest+'.f32');assert len(raw)==ROW_BYTES and sha(raw)==digest;cache[digest]=raw
            start=spec['visual_starts'][b]
            if start>=0 and start<=t<start+64:
                key=(spec['visual_epoch'],spec['row_ids'][b],t-start)
                if key not in visuals:visuals[key]=visual_row_bytes(*key)
                want=visuals[key]
            else:want=oracle[selected[b][t] if t<len(selected[b]) else 0]
            assert cache[digest]==want,('Native prompt byte mismatch',spec['key'],kind,t,b)
            logical_bytes+=ROW_BYTES
    stats=record['cache_stats'];assert stats['physical_batch']==spec['batch'] and stats['tokens']==max(spec['lengths']) and stats['lengths']==spec['lengths'] and stats['page_size']==spec['page_size']
    assert len(stats['cached_pages'])<=2 and 0<stats['peak_cached_packed_logical_bytes']<=2*spec['page_size']*spec['batch']*ROW_BYTES
    assert record['post_close_rejected'] is True
    assert type(record['first_page_evicted']) is bool and record['first_page_evicted']==(max(spec['lengths'])>2*spec['page_size'])
    return {'key':spec['key'],'read_events':len(events),'logical_bytes_reconstructed_and_verified':logical_bytes,'unique_actual_row_blobs':len(cache),
            'raw_row_sha256':sorted(cache),'all_observed_bytes_equal_independent_oracle':True,'performance_admitted':False}
