"""Read one existing BF16 carrier after each ordinary decode step; no graph edits."""
import re,struct
from .b4_k35_cache52 import key_layout,checked_cache,analyze as primary_analysis
COORDINATE={'layer':35,'head':0,'slot':4012,'dimension':211}
PAIR_WORD=(4012*256+211)//2
assert PAIR_WORD==513641

def codes(raw):
    assert isinstance(raw,bytes) and len(raw)==4
    pair=struct.unpack('<HH',raw);assert all((v&0x7f80)!=0x7f80 for v in pair),'Nonfinite BF16 coordinate or neighbor'
    return list(pair)

class CoordinateSampler:
    def __init__(self,io,layout,batch,position):
        assert type(batch) is type(position) is int and (batch,position) in ((1,0),(4,2))
        spec=key_layout(layout);self.io=io;self.batch=batch;self.position=position;self.offset=spec['offset']+PAIR_WORD
        self.part=io.zeros([batch,1]);self.points=[]
    def read(self,state,step,slots):
        assert type(step) is int and step==len(self.points) and 0<=step<=52 and slots==[3968+step]*self.batch
        values=[]
        for repeat in range(2):
            self.io.slice_into(state,self.offset,self.part);values.append(bytes(self.io.finite_f32_bytes(self.io.row(self.part,self.position))))
        assert len(values[0])==4 and values[0]==values[1],'Repeated carrier reads changed'
        pair=codes(values[0]);point={'step':step,'slots':list(slots),'carrier_hex':values[0].hex(),'repeat_carrier_hex':[v.hex() for v in values],'BF16_codes':pair,'selected_BF16_bits':pair[1]}
        self.points.append(point);return point
    def result(self):
        assert len(self.points)==53
        return {'coordinate':dict(COORDINATE),'carrier_word_in_key_cache':PAIR_WORD,'carrier_word_in_physical_state':self.offset,'physical_batch':self.batch,'physical_row':self.position,'selected_halfword':1,'neighbor_dimension':210,'points':self.points}

def checked_trajectory(value,raw_cache,batch,position):
    assert (batch,position) in ((1,0),(4,2))
    expected={'coordinate':COORDINATE,'carrier_word_in_key_cache':PAIR_WORD,'carrier_word_in_physical_state':55894016+PAIR_WORD,'physical_batch':batch,'physical_row':position,'selected_halfword':1,'neighbor_dimension':210}
    assert set(value)==set(expected)|{'points'}
    for k,v in expected.items():assert value[k]==v,(k,value[k],v)
    points=value['points'];assert len(points)==53 and [v['step'] for v in points]==list(range(53))
    for step,p in enumerate(points):
        assert set(p)=={'step','slots','carrier_hex','repeat_carrier_hex','BF16_codes','selected_BF16_bits'} and type(p['step']) is int and p['slots']==[3968+step]*batch
        assert isinstance(p['carrier_hex'],str) and re.fullmatch('[0-9a-f]{8}',p['carrier_hex'])
        raw=bytes.fromhex(p['carrier_hex']);assert p['repeat_carrier_hex']==[p['carrier_hex']]*2 and p['BF16_codes']==codes(raw) and p['selected_BF16_bits']==p['BF16_codes'][1]
    assert bytes.fromhex(points[-1]['carrier_hex'])==raw_cache[PAIR_WORD*4:PAIR_WORD*4+4],'Step52 sample must link to complete8MiB cache'
    return points

def compare(reference,candidate):
    assert len(reference)==len(candidate)==53;rows=[]
    for step,(a,b) in enumerate(zip(reference,candidate)):
        assert a['step']==b['step']==step
        x,y=a['selected_BF16_bits'],b['selected_BF16_bits']
        rows.append({'step':step,'reference_carrier_hex':a['carrier_hex'],'candidate_carrier_hex':b['carrier_hex'],'carrier_exact':a['carrier_hex']==b['carrier_hex'],'selected_bits_exact':x==y,'reference_BF16_bits':x,'candidate_BF16_bits':y,'reference_value':struct.unpack('<f',struct.pack('<I',x<<16))[0],'candidate_value':struct.unpack('<f',struct.pack('<I',y<<16))[0]})
    differing=[r['step'] for r in rows if not r['selected_bits_exact']];carrier=[r['step'] for r in rows if not r['carrier_exact']]
    return {'status':'ordinary_key35_coordinate_53_timepoints_compared','coordinate':dict(COORDINATE),'first_observed_selected_value_difference_step':min(differing) if differing else None,'first_observed_carrier_difference_step':min(carrier) if carrier else None,'selected_value_differing_steps':differing,'scheduled_decode_writer_step':45,'rows':rows,'scope':'Observed after completed ordinary decoder steps with repeated GPU reads. Only this coordinate/carrier is sampled between full primary checkpoints; not the first divergence of the entire decoder or proof of a unique causal operator.'}

def analyze(selection,key,reference,candidate,old_reference,old_candidate,read_reference,read_candidate,read_old_reference,read_old_candidate):
    ordinary=primary_analysis(selection,key,reference,candidate,old_reference,old_candidate,lambda k,n:read_reference(n),read_candidate,lambda k,n:read_old_reference(n),read_old_candidate)
    reports=(reference,candidate);reads=(read_reference,read_candidate);batches=(1,4);positions=(0,2);points=[]
    for report,read,batch,position in zip(reports,reads,batches,positions):
        row=report['rows'][0];raw=checked_cache(row['key35_cache52'],row['final_state']['logical']['components'][70],batch,position,read)
        parent=old_reference if batch==1 else old_candidate;old_read=read_old_reference if batch==1 else read_old_candidate
        old_row=parent['rows'][0];oldraw=checked_cache(old_row['key35_cache52'],old_row['final_state']['logical']['components'][70],batch,position,old_read)
        assert raw==oldraw,'Full cache must repeat same-profile ordinary parent'
        points.append(checked_trajectory(row['key35_coordinate_trajectory'],raw,batch,position))
    return {'status':'ordinary_key35_coordinate_trajectory_after_complete_primary_and_raw_cache_parity','ordinary_primary_and_cache':ordinary,'trajectory':compare(*points),'model_or_full4K_MTP_quality_HTTP_or_performance_qualified':False,'original_full4k_still_failed':True,'two_times_target_achieved':False}
