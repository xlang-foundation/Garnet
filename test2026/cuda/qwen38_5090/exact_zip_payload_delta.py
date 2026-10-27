"""Lossless ZIP transport: reuse byte-identical compressed payloads, preserve all headers.

This transports the exact archive bytes. It does not alter reports, arrays, or gates.
"""
from pathlib import Path
import argparse,hashlib,json,os,struct,zipfile
FORMAT='garnet-exact-zip-payload-delta-v1'
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()
def region(f,offset,length):
    f.seek(offset);raw=f.read(length);assert len(raw)==length;return raw
def payload(f,info):
    header=region(f,info.header_offset,30);v=struct.unpack('<IHHHHHIIIHH',header)
    assert v[0]==0x04034b50 and not (v[2]&1)
    return info.header_offset+30+v[-2]+v[-1],info.compress_size
def build(source,target,delta,expected):
    source,target,delta=map(Path,(source,target,delta));assert source!=target and not delta.exists() and digest(target)==expected
    source_sha=digest(source);index={};matches=[]
    with source.open('rb') as old,target.open('rb') as new,zipfile.ZipFile(source) as oldzip,zipfile.ZipFile(target) as newzip:
        for info in oldzip.infolist():
            offset,length=payload(old,info)
            if length>=1024:
                raw=region(old,offset,length);index.setdefault((length,hashlib.sha256(raw).hexdigest()),(offset,info.filename))
        for info in newzip.infolist():
            offset,length=payload(new,info)
            if length<1024:continue
            raw=region(new,offset,length);h=hashlib.sha256(raw).hexdigest();found=index.get((length,h))
            if found:
                assert raw==region(old,found[0],length)
                matches.append(dict(target_offset=offset,bytes=length,kind='parent',source_offset=found[0],source_member=found[1],sha256=h))
        matches.sort(key=lambda r:r['target_offset']);segments=[];inline=bytearray();cursor=0
        def gap(end):
            nonlocal cursor
            assert end>=cursor
            if end>cursor:
                raw=region(new,cursor,end-cursor);segments.append(dict(target_offset=cursor,bytes=len(raw),kind='inline',source_offset=len(inline),sha256=hashlib.sha256(raw).hexdigest()));inline.extend(raw)
            cursor=end
        for match in matches:
            gap(match['target_offset']);segments.append(match);cursor+=match['bytes']
        gap(target.stat().st_size)
    manifest=dict(format=FORMAT,source_name=source.name,source_sha256=source_sha,source_bytes=source.stat().st_size,target_name=target.name,target_sha256=expected,target_bytes=target.stat().st_size,segments=segments,inline_bytes=len(inline),inline_sha256=hashlib.sha256(inline).hexdigest(),identical_compressed_payloads=len(matches),reused_payload_bytes=sum(r['bytes'] for r in matches),model_or_numerical_gates_changed=False)
    with zipfile.ZipFile(delta,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('delta-manifest.json',json.dumps(manifest,indent=2));z.writestr('inline.bin',inline)
    return dict(status='exact_archive_delta_built',delta_sha256=digest(delta),delta_bytes=delta.stat().st_size,target_sha256=expected,target_bytes=manifest['target_bytes'],reused_payload_bytes=manifest['reused_payload_bytes'],identical_compressed_payloads=len(matches))
def apply(source,delta,target,expected):
    source,delta,target=map(Path,(source,delta,target));assert not target.exists()
    partial=target.with_name(target.name+'.delta-part');assert not partial.exists()
    with zipfile.ZipFile(delta) as z:
        assert z.testzip() is None and set(z.namelist())=={'delta-manifest.json','inline.bin'} and len(z.namelist())==2
        m=json.loads(z.read('delta-manifest.json'));inline=z.read('inline.bin')
    assert m['format']==FORMAT and m['model_or_numerical_gates_changed'] is False
    assert m['source_name']==source.name and m['target_name']==target.name and m['target_sha256']==expected
    assert digest(source)==m['source_sha256'] and source.stat().st_size==m['source_bytes']
    assert len(inline)==m['inline_bytes'] and hashlib.sha256(inline).hexdigest()==m['inline_sha256']
    cursor=0;whole=hashlib.sha256()
    with source.open('rb') as old,partial.open('xb') as out:
        for s in m['segments']:
            assert type(s['target_offset']) is type(s['source_offset']) is type(s['bytes']) is int and s['target_offset']==cursor and s['source_offset']>=0 and s['bytes']>0
            assert cursor+s['bytes']<=m['target_bytes']
            if s['kind']=='parent':
                assert s['source_offset']+s['bytes']<=m['source_bytes'];raw=region(old,s['source_offset'],s['bytes'])
            else:
                assert s['kind']=='inline' and s['source_offset']+s['bytes']<=len(inline);raw=inline[s['source_offset']:s['source_offset']+s['bytes']]
            assert hashlib.sha256(raw).hexdigest()==s['sha256'];out.write(raw);whole.update(raw);cursor+=len(raw)
    assert cursor==m['target_bytes']==partial.stat().st_size and whole.hexdigest()==expected and not target.exists()
    os.replace(partial,target)
    return dict(status='original_archive_exact_bytes_reconstructed',archive_sha256=expected,archive_bytes=cursor,source_sha256=m['source_sha256'],delta_sha256=digest(delta),identical_compressed_payloads=m['identical_compressed_payloads'],numerical_or_oracle_or_archive_byte_change=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('build','apply'));p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--delta',type=Path,required=True);p.add_argument('--expected-target-sha',required=True);a=p.parse_args()
    result=build(a.source,a.target,a.delta,a.expected_target_sha) if a.mode=='build' else apply(a.source,a.delta,a.target,a.expected_target_sha)
    print(json.dumps(result))
if __name__=='__main__':main()
