"""Independent integer byte oracle for the isolated B1 cache commit candidate."""
import struct
if __package__:
 from .bf16_low_batch_layout import build_bf16_low_batch_layout,bf16_low_batch_byte_offset
else:
 from bf16_low_batch_layout import build_bf16_low_batch_layout,bf16_low_batch_byte_offset

def case_plan():
 result=[]
 for chunk in (4,32):
  def add(source,trial,phase,start,count):result.append(dict(id=len(result),batch=1,chunk=chunk,source=source,trial=trial,phase=phase,start=start,count=count))
  add(0,-1,'zero',0,0)
  counts=(0,1,2,3,4) if chunk==4 else (0,1,16,31,32)
  starts=(0,511,512,513,4096-chunk)
  for trial,count in enumerate(counts):
   for source in range(2 if chunk==4 else 1):add(source,trial,'trial',starts[(2*trial)%5] if count else 4096,count)
  for source in range(2 if chunk==4 else 1):add(source,-1,'graph',4096-chunk,chunk)
 return result

def pattern(i,salt):return 0x3f000000|((i*97+salt)&0x7fffff)|(0x80000000 if i&1 else 0)
def rounded_bf16(bits):return ((bits+0x7fff+((bits>>16)&1))>>16)&0xffff

def samples(case):
 wanted={'id','batch','chunk','source','trial','phase','start','count'}
 if set(case)!=wanted or any(type(case[k]) is not int for k in wanted-{'phase'}) or case not in case_plan():raise ValueError('Undeclared B1 commit case')
 layout=build_bf16_low_batch_layout(chunk=case['chunk'],retain_linear_inputs=case['chunk']==4)
 start,count=case['start'],case['count'];slots={0,4095}
 if start:slots.add(start-1)
 if start<4096:slots.add(start)
 if count:slots.add(start+count-1)
 if start+count<4096:slots.add(start+count)
 result=[]
 for layer in layout['layers']:
  if layer['kind']=='linear_attention':
   width=layer['state1_width']+layer['state2_width']
   for i in (0,width//2,width-1):
    origin=layer['replay_offset'] if case['source'] else layer['target_offset']
    bits=pattern(origin+i,811 if case['source'] else 19) if count else 0xd3d3d3d3
    result.append({'layer':layer['index'],'kind':'linear','canonical_byte':4*(layer['offset']+i),'bytes':4,'bits':bits})
  else:
   for which in ('key','value'):
    for head in range(4):
     for slot in sorted(slots):
      for dimension in (0,1,128,255):
       bits=0xd3d3
       if start<=slot<start+count:
        origin=layer['target_offset']+(case['chunk']*1024 if which=='value' else 0)+((slot-start)*4+head)*256+dimension
        bits=rounded_bf16(pattern(origin,19))
       result.append({'layer':layer['index'],'kind':which,'canonical_byte':bf16_low_batch_byte_offset(layout,layer['index'],which,0,head,slot,dimension),'bytes':2,'bits':bits})
 return result

def expected(case):return b''.join(struct.pack('<I' if s['bytes']==4 else '<H',s['bits']) for s in samples(case))

def compare(case,raw):
 wanted=expected(case)
 if raw!=wanted:
  first=next((i for i,(a,b) in enumerate(zip(raw,wanted)) if a!=b),min(len(raw),len(wanted)))
  raise ValueError('Exact commit bytes differ: case='+str(case['id'])+',first_byte='+str(first))
 return {'id':case['id'],'bytes':len(raw),'samples':len(samples(case)),'exact':True}
