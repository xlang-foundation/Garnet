"""Original SDK log parser adapted only to two FA value cases and one matrix each."""
import re
PROJECTIONS={'v3':1024,'v11':1024}

def parse_log(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8')
    assert 'CUBLASLT_VERSION 130400' in raw and 'SM120' in raw and 'ACTUAL_M4_M16_SDK_CAPTURE_COMPLETE_NO_MODEL_OR_PERFORMANCE_ADMISSION' in raw
    pattern=r'ALGO projection=(v3|v11) step=(\d+) repeat=(\d+) row=(\d+) mode=(m4|default16|shared16) m=(\d+) status=(-?\d+) id=(-?\d+) tile=(-?\d+) stages=(-?\d+) split=(-?\d+) reduction=(-?\d+) swizzle=(-?\d+) custom=(-?\d+) workspace=(\d+)'
    calls=[];configs={};seen=set()
    for match in re.finditer(pattern,raw):
        p,s,r,row,mode,m,status,*rest=match.groups();s,r,row,m,status=map(int,(s,r,row,m,status));values=list(map(int,rest));attrs=values[:7];workspace=values[7]
        assert s in range(1) and r in (0,1) and status==0 and workspace==4<<20 and m==(4 if mode=='m4' else 16) and row in (range(4) if mode=='m4' else range(1))
        key=(p,s,r,row,mode);assert key not in seen;seen.add(key)
        if mode=='m4':
            if p in configs:assert configs[p]==attrs
            else:configs[p]=attrs
        if mode=='shared16':assert configs[p]==attrs
        calls.append(dict(projection=p,step=s,repeat=r,row=row,mode=mode,M=m,attributes=dict(zip(('id','tile','stages','split','reduction','swizzle','custom'),attrs)),workspace=workspace))
    assert len(calls)==24 and set(configs)==set(PROJECTIONS)
    checks=re.findall(r'ALGO_CHECK projection=(v3|v11) step=(\d+) repeat=(\d+) m=(\d+) status=(-?\d+) state=(-?\d+) workspace=(\d+) capacity=(\d+)',raw);assert len(checks)==4
    checkkeys=set()
    for p,s,r,m,status,state,workspace,capacity in checks:
        assert (p,int(s),int(r)) not in checkkeys;checkkeys.add((p,int(s),int(r)));assert int(s) in range(1) and int(r) in (0,1) and int(m)==16 and int(status)==int(state)==0 and int(workspace)<=int(capacity)==4<<20
    return dict(status='actual_M4_M16_SDK_attributes_and_AlgoCheck_replayed',calls=calls,actual_M16_AlgoChecks=4,reference_algorithms=configs,all_seven_attributes_queried=True,no_unsupported_or_workspace_fallback=True)
