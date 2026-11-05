"""Extract only the pinned parent's MTP tensors through one validated HTTP range.

Consumes the previously frozen parent index/header audit. Preserves raw BF16
bits in a standalone safetensors file; does not alter or qualify the generator.
Never run during a model timing/quality campaign or another inference process.
"""
import argparse,hashlib,json,math,os,re,shutil,struct,urllib.request
from pathlib import Path
import numpy as np
from run_local_coder_baseline import engine_processes

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();live=engine_processes()
    if live:raise RuntimeError('Another inference engine is live: '+str(live))
    root=args.metadata_root.resolve();audit=json.loads((root/'audit.json').read_text())
    header_audit=json.loads((root/'tensor-header-audit.json').read_text())
    revision=audit['revision'];filename=header_audit['file']
    if audit['repo_id']!='Qwen/Qwen3.8-27B' or not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Unexpected parent identity')
    if header_audit['status']!='header_read' or header_audit['revision']!=revision or not re.fullmatch(r'model-[0-9]+-of-[0-9]+\.safetensors',filename):raise ValueError('Parent header audit differs')
    if digest(root/'model.safetensors.index.json')!=audit['files']['model.safetensors.index.json']['sha256']:raise ValueError('Frozen parent index changed')
    if digest(root/'last-shard-header.json')!=header_audit['ranges'][1]['sha256']:raise ValueError('Frozen parent header changed')
    header=json.loads((root/'last-shard-header.json').read_text());index=json.loads((root/'model.safetensors.index.json').read_text())
    keys=audit['matching_keys']
    if len(keys)!=15 or any(not key.startswith('mtp.') or index['weight_map'][key]!=filename for key in keys):raise ValueError('Unexpected MTP namespace or shard layout')
    tensors={key:header[key] for key in keys};ordered=sorted(tensors,key=lambda key:tensors[key]['data_offsets'][0])
    first=tensors[ordered[0]]['data_offsets'][0];cursor=first
    for key in ordered:
        value=tensors[key];start,end=value['data_offsets']
        if value['dtype']!='BF16' or any(type(n) is not int or n<=0 for n in value['shape']) or start!=cursor or end-start!=2*math.prod(value['shape']):raise ValueError('Noncontiguous or invalid MTP tensor')
        cursor=end
    payload_bytes=cursor-first
    total=int(header_audit['ranges'][0]['content_range'].rsplit('/',1)[1]);data_start=8+(root/'last-shard-header.json').stat().st_size
    start=data_start+first;end=data_start+cursor-1
    if end>=total or payload_bytes!=header_audit['total_payload_bytes']:raise ValueError('Parent payload range differs')
    if shutil.disk_usage(args.output.parent).free<payload_bytes+(1<<30):raise RuntimeError('Insufficient free disk for bounded MTP extraction')
    args.output.mkdir(exist_ok=False);out=args.output.resolve()
    package_header={key:dict(dtype='BF16',shape=value['shape'],data_offsets=[x-first for x in value['data_offsets']]) for key,value in tensors.items()}
    package_header['__metadata__']={'format':'pt','source_repo':audit['repo_id'],'source_revision':revision,'scope':'Raw auxiliary MTP tensors only'}
    encoded=json.dumps(package_header,separators=(',',':'),sort_keys=True).encode();encoded+=b' '*((-len(encoded))%8)
    prefix=struct.pack('<Q',len(encoded))+encoded
    report=dict(status='downloading',scope=__doc__,repo_id=audit['repo_id'],revision=revision,parent_file=filename,
        parent_header_sha256=digest(root/'last-shard-header.json'),parent_index_sha256=digest(root/'model.safetensors.index.json'),
        payload_bytes=payload_bytes,range_start=start,range_end=end,source_total_bytes=total,downloaded_bytes=0,
        base_generator_modified=False,source_sha256=digest(Path(__file__)),tensors=tensors)
    def save():(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    save();shutil.copy2(Path(__file__),out/'fetch-source.py')
    for name in ['audit.json','tensor-header-audit.json','last-shard-header.json','model.safetensors.index.json']:shutil.copy2(root/name,out/('parent-'+name))
    temporary=out/'mtp-bf16.safetensors.part';target=out/'mtp-bf16.safetensors'
    try:
        request=urllib.request.Request('https://huggingface.co/'+audit['repo_id']+'/resolve/'+revision+'/'+filename,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=30) as response,temporary.open('xb') as stream:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':raise ValueError('Server did not honor the exact auxiliary-only range')
            if response.headers.get('Content-Length') is not None and int(response.headers['Content-Length'])!=payload_bytes:raise ValueError('Range content length differs')
            stream.write(prefix);whole=hashlib.sha256(prefix);payload=hashlib.sha256();next_progress=64<<20
            while report['downloaded_bytes']<payload_bytes:
                block=response.read(min(8<<20,payload_bytes-report['downloaded_bytes']))
                if not block:raise ValueError('Truncated MTP range response')
                stream.write(block);whole.update(block);payload.update(block);report['downloaded_bytes']+=len(block)
                if report['downloaded_bytes']>=next_progress:
                    save();print(json.dumps(dict(downloaded_bytes=report['downloaded_bytes'],total_bytes=payload_bytes)),flush=True);next_progress+=64<<20
            if response.read(1):raise ValueError('Unexpected trailing range bytes')
            stream.flush();os.fsync(stream.fileno())
        if temporary.stat().st_size!=len(prefix)+payload_bytes:raise ValueError('Extracted package size differs')
        values=np.memmap(temporary,dtype='<u2',mode='r',offset=len(prefix),shape=(payload_bytes//2,))
        for offset in range(0,len(values),8<<20):
            if np.any((values[offset:offset+(8<<20)]&0x7f80)==0x7f80):raise ValueError('Nonfinite raw BF16 MTP tensor')
        del values
        temporary.rename(target)
        (out/'model.safetensors.index.json').write_text(json.dumps(dict(metadata={'total_size':payload_bytes},weight_map={key:target.name for key in keys}),indent=2)+'\n')
        report.update(status='extracted_raw_bf16_not_runtime_qualified',payload_sha256=payload.hexdigest(),package_sha256=whole.hexdigest(),package_bytes=target.stat().st_size,all_bf16_values_finite=True)
        save();print(json.dumps(dict(status=report['status'],package_bytes=report['package_bytes'],tensors=len(keys))),flush=True)
    except Exception as error:report.update(status='failed',error=repr(error));save();raise
if __name__=='__main__':main()
