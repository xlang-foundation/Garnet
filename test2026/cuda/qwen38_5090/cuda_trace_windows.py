"""Conservative selection of kernel records between unique HtoD markers."""

def select_window(kernels,copies,start_bytes,end_bytes):
    starts=[r for r in copies if r['copyKind']==1 and r['bytes']==start_bytes]
    ends=[r for r in copies if r['copyKind']==1 and r['bytes']==end_bytes]
    assert len(starts)==len(ends)==1,'Unique HtoD markers required'
    start,end=starts[0],ends[0];left,right=start['end'],end['start']
    assert left<right and start['globalPid']==end['globalPid'] and start['deviceId']==end['deviceId']
    matching=[k for k in kernels if k['globalPid']==start['globalPid'] and k['deviceId']==start['deviceId']]
    assert not [k for k in matching if k['start']<right and k['end']>left and not(k['start']>=left and k['end']<=right)],'Kernel crosses marker boundary'
    selected=[k for k in matching if k['start']>=left and k['end']<=right]
    assert all(k['end']>=k['start'] for k in selected)
    return left,right,selected
