"""Strict selected-row comparison of complete packed BF16 B1/B4 witnesses.

Separate from the packed-versus-FP32-reference predicate. Both inputs here
must retain their actual packed representation flags and physical widths.
"""
import re

def _validate(value):
 if not isinstance(value,dict) or type(value.get('physical_words_per_row')) is not int or value['physical_words_per_row']!=106823680 or type(value.get('logical_words_per_row')) is not int or value['logical_words_per_row']!=173932544 or value.get('reference_bf16_roundtrip_checked') is not False:
  raise ValueError('Expected a complete packed BF16 state witness')
 slots=value.get('slots');components=value.get('components')
 if not isinstance(slots,list) or len(slots) not in (1,4) or any(type(n) is not int or not 0<=n<=4096 for n in slots):raise ValueError('Invalid packed B1/B4 slots')
 batch=len(slots)
 def hashes(rows):
  if not isinstance(rows,list) or len(rows)!=batch or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in rows):raise ValueError('Invalid complete packed state hashes')
 hashes(value.get('physical_row_sha256'));hashes(value.get('logical_row_sha256'))
 if not isinstance(components,list) or len(components)!=128:raise ValueError('Missing packed state components')
 for i,c in enumerate(components):
  full=i//2%4==3;words=4194304 if full else (40960 if i%2==0 else 786432)
  if not isinstance(c,dict) or type(c.get('layer')) is not int or c['layer']!=i//2 or type(c.get('role')) is not int or c['role']!=i%2+1 or c.get('kind')!=('full_attention' if full else 'linear_attention') or type(c.get('logical_words')) is not int or c['logical_words']!=words:raise ValueError('Packed component geometry/order mismatch')
  hashes(c.get('row_sha256'))
 if sum(c['logical_words'] for c in components)!=173932544:raise ValueError('Incomplete packed state width')
 return batch

def same_packed_bf16_row(left,row,right,right_row):
 left_batch=_validate(left);right_batch=_validate(right)
 if type(row) is not int or type(right_row) is not int or not 0<=row<left_batch or not 0<=right_row<right_batch:raise ValueError('Selected packed state row outside profile')
 return all(left[k][row]==right[k][right_row] for k in ('slots','physical_row_sha256','logical_row_sha256')) and all(a['row_sha256'][row]==b['row_sha256'][right_row] for a,b in zip(left['components'],right['components']))
