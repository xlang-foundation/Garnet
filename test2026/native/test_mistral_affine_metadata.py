#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small real-format checkpoint fixtures for native representation guards."""
import argparse,copy,json,pathlib,struct,subprocess,hashlib

def save(folder,config,tensors):
 folder.mkdir();header={};data=b''
 for name,kind,shape,payload in tensors:
  header[name]=dict(dtype=kind,shape=shape,data_offsets=[len(data),len(data)+len(payload)]);data+=payload
 wire=json.dumps(header,separators=(',',':')).encode();wire+=b' '*((-len(wire))%8)
 (folder/'model.safetensors').write_bytes(struct.pack('<Q',len(wire))+wire+data)
 (folder/'config.json').write_text(json.dumps(config)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=pathlib.Path,required=True)
 p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 config=dict(model_type='mistral',quantization=dict(bits=4,group_size=64))
 tensors=[('model.embed_tokens.weight','U32',[1,8],bytes(32)),('model.embed_tokens.scales','F16',[1,1],struct.pack('<H',0x3c00)),('model.embed_tokens.biases','F16',[1,1],bytes(2))]
 scenarios=[]
 def case(name,mutator,accepted=False):
  c=copy.deepcopy(config);t=copy.deepcopy(tensors);mutator(c,t);scenarios.append((name,c,t,accepted))
 case('fp16-valid',lambda c,t:None,True)
 case('bf16-valid',lambda c,t:t.__setitem__(slice(1,3),[(t[1][0],'BF16',[1,1],struct.pack('<H',0x3f80)),(t[2][0],'BF16',[1,1],bytes(2))]),True)
 case('mixed-coefficient-types',lambda c,t:t.__setitem__(2,(t[2][0],'BF16',[1,1],bytes(2))))
 case('unsupported-coefficient-type',lambda c,t:t.__setitem__(1,(t[1][0],'F32',[1,1],struct.pack('<f',1.))))
 case('missing-bias',lambda c,t:t.pop())
 case('wrong-scale-shape',lambda c,t:t.__setitem__(1,(t[1][0],'F16',[2,1],bytes(4))))
 case('wrong-bias-shape',lambda c,t:t.__setitem__(2,(t[2][0],'F16',[1,2],bytes(4))))
 case('wrong-bits',lambda c,t:c['quantization'].update(bits=8))
 case('wrong-group',lambda c,t:c['quantization'].update(group_size=32))
 case('wrong-mode',lambda c,t:c['quantization'].update(mode='mxfp4'))
 case('unqualified-family',lambda c,t:c.update(model_type='llama'))
 case('ambiguous-roots',lambda c,t:t.append(('model.language_model.embed_tokens.weight','U32',[1,8],bytes(32))))
 request=a.output/'request.json';request.write_text(json.dumps([dict(tensor='model.language_model.embed_tokens.weight',rows=[0])])+'\n');rows=[]
 for name,c,t,accepted in scenarios:
  folder=a.output/name;save(folder,c,t)
  result=subprocess.run([str(a.binary),str(folder),str(request),str(folder/'decoded')],capture_output=True,text=True,timeout=30)
  ok=result.returncode==0 if accepted else result.returncode==1 and bool(result.stderr)
  rows.append(dict(case=name,expected_accepted=accepted,returncode=result.returncode,error=result.stderr.strip(),pass_case=ok))
 report=dict(passed=all(x['pass_case'] for x in rows),cases=rows,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),test_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
