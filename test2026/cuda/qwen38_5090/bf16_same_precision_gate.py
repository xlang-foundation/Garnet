"""Compare complete logical state across two physical layouts; no GPU imports."""
import re


def same_logical_state(packed,reference):
    for value,physical,roundtrip in ((packed,106823680,False),(reference,173932544,True)):
        if type(value.get('physical_words_per_row')) is not int or value['physical_words_per_row']!=physical or type(value.get('logical_words_per_row')) is not int or value['logical_words_per_row']!=173932544 or value.get('reference_bf16_roundtrip_checked') is not roundtrip:
            raise ValueError('Unqualified same-precision state representation')
        slots=value.get('slots');components=value.get('components')
        if not isinstance(slots,list) or len(slots)!=4 or any(type(n) is not int or not 0<=n<=4096 for n in slots):raise ValueError('Invalid complete state slots')
        if not isinstance(components,list) or len(components)!=128:raise ValueError('Missing complete logical state components')
        def hashes(rows):
            if not isinstance(rows,list) or len(rows)!=4 or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in rows):raise ValueError('Invalid state hash witnesses')
        hashes(value.get('physical_row_sha256'));hashes(value.get('logical_row_sha256'))
        if roundtrip and value['physical_row_sha256']!=value['logical_row_sha256']:raise ValueError('Reference physical/logical bytes must agree')
        for i,component in enumerate(components):
            full=i//2%4==3;words=4194304 if full else (40960 if i%2==0 else 786432)
            if type(component.get('layer')) is not int or component['layer']!=i//2 or type(component.get('role')) is not int or component['role']!=i%2+1 or component.get('kind')!=('full_attention' if full else 'linear_attention') or type(component.get('logical_words')) is not int or component['logical_words']!=words:
                raise ValueError('Logical component geometry/order mismatch')
            hashes(component.get('row_sha256'))
        if sum(c['logical_words'] for c in components)!=173932544:raise ValueError('Incomplete logical state width')
    return packed['slots']==reference['slots'] and packed['logical_row_sha256']==reference['logical_row_sha256'] and packed['components']==reference['components']
