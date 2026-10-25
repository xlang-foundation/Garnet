"""Independent dense FP64 reference reused for a separately declared B1 plan."""
from bf16_kv_overlay_oracle import ATOL,RTOL,compare_witness,sample_triples,reference_case,spans

def case_plan():
    cases=[{'batch':1,'chunk':chunk,'layer':layer,'end':end}
           for chunk in (4,32) for layer in (3,63) for end in (0,1,511,512,513,4095,4096)]
    cases += [{'batch':1,'chunk':4,'layer':layer,'end':513} for layer in range(7,63,4)]
    result=[dict(case,id=i) for i,case in enumerate(cases)]
    # The unchanged mask generator uses1+id%chunk for row0. IDs63/95
    # therefore exercise all32tokens at the first/last cache boundaries.
    result += [dict(batch=1,chunk=32,layer=3,end=32,id=63),dict(batch=1,chunk=32,layer=63,end=4096,id=95)]
    return result
