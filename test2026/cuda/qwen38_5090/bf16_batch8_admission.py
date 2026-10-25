"""Pure row-permutation controls for the existing, still-unqualified B8 profile."""
from .bf16_packed_state_gate import same_packed_bf16_row

ORDERS=(tuple(range(8)),(3,4,5,6,7,0,1,2),tuple(range(8)))

def _selected(witness,row):
    """Select all components, retaining the old strict B1 geometry validator."""
    if type(row) is not int or not 0<=row<8:raise ValueError('A B8 physical row is required')
    result=dict(witness)
    for key in ('slots','physical_row_sha256','logical_row_sha256'):
        value=witness.get(key)
        if not isinstance(value,list) or len(value)!=8:raise ValueError('Complete B8 state rows required')
        result[key]=[value[row]]
    components=witness.get('components')
    if not isinstance(components,list) or len(components)!=128:raise ValueError('Complete B8 state components required')
    result['components']=[]
    for component in components:
        hashes=component.get('row_sha256')
        if not isinstance(hashes,list) or len(hashes)!=8:raise ValueError('Complete B8 component rows required')
        result['components'].append(dict(component,row_sha256=[hashes[row]]))
    # This validates every width, dtype/representation flag, component and hash.
    if not same_packed_bf16_row(result,0,result,0):raise ValueError('Invalid selected state')
    return result

def one_row(witness,row):
    if type(row) is not int or not 0<=row<8:raise ValueError('A B8 physical row is required')
    rows=[_selected(witness,i) for i in range(8)]
    return rows[row]

def compare_permuted(left,right,order):
    if not isinstance(order,(list,tuple)) or len(order)!=8 or any(type(v) is not int for v in order) or sorted(order)!=list(range(8)):
        raise ValueError('An exact permutation of eight rows is required')
    left_rows=[_selected(left,i) for i in range(8)];right_rows=[_selected(right,i) for i in range(8)]
    return all(same_packed_bf16_row(left_rows[original],0,right_rows[physical],0) for physical,original in enumerate(order))

def compare_vector_rows(left,right,order):
    if not isinstance(left,(bytes,bytearray)) or not isinstance(right,(bytes,bytearray)) or len(left)!=8*5120*4 or len(right)!=len(left):raise ValueError('Eight complete FP32 hidden rows required')
    if not isinstance(order,(list,tuple)) or len(order)!=8 or any(type(v) is not int for v in order) or sorted(order)!=list(range(8)):raise ValueError('Exact row permutation required')
    width=5120*4
    return all(left[original*width:(original+1)*width]==right[physical*width:(physical+1)*width] for physical,original in enumerate(order))
