"""Acquire only pinned validation Parquets, with immutable blob/hash checks."""
from pathlib import Path,PurePosixPath
import argparse,concurrent.futures,datetime,hashlib,json,re,shutil,urllib.request


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    root=a.root.resolve(strict=True);api=root/'dataset-api-main.json';tree=root/'dataset-tree.json'
    info=json.loads(api.read_bytes());entries=json.loads(tree.read_bytes())
    assert info['id']=='MMMU/MMMU' and info['sha']=='876ce5cb130f7f7e290ce4d9984357737d4db5cf'
    files=[item for item in entries if item['type']=='file' and '/validation-' in item['path']]
    assert len(files)==30 and sum(item['size'] for item in files)==341234748
    assert shutil.disk_usage(root).free>=sum(item['size'] for item in files)+2*1024**3
    target=root/'parquet';target.mkdir(exist_ok=True);resolved=target.resolve(strict=True)
    assert resolved.parent==root
    report=root/'validation-acquisition.json';events=root/'validation-download-events.jsonl';results=[];errors=[]
    def acquire(item):
        name=item['path'];assert re.fullmatch(r'[A-Za-z_]+/validation-00000-of-00001\.parquet',name)
        assert len(PurePosixPath(name).parts)==2 and re.fullmatch('[0-9a-f]{64}',item['lfs']['oid']) and item['lfs']['size']==item['size']
        subject=name.split('/')[0];dest=target/(subject+'.parquet');part=target/(subject+'.parquet.part')
        assert dest.resolve().parent==resolved and part.resolve().parent==resolved
        url='https://huggingface.co/datasets/MMMU/MMMU/resolve/'+info['sha']+'/'+name
        if dest.exists():
            assert dest.is_file() and dest.stat().st_size==item['size'] and sha(dest)==item['lfs']['oid']
            return {'subject':subject,'source_path':name,'file':dest.relative_to(root).as_posix(),'bytes':item['size'],'sha256':item['lfs']['oid'],'url':url,'status':'reused_hash_verified'}
        offset=part.stat().st_size if part.exists() else 0
        assert 0<=offset<=item['size']
        if offset<item['size']:
            request=urllib.request.Request(url,headers={'User-Agent':'CantorAI-Garnet-research/1'})
            if offset:request.add_header('Range','bytes='+str(offset)+'-')
            with urllib.request.urlopen(request,timeout=90) as response:
                if offset:assert response.status==206 and response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')
                with part.open('ab' if offset else 'xb') as stream:
                    while True:
                        block=response.read(1<<20)
                        if not block:break
                        stream.write(block)
        assert part.stat().st_size==item['size'] and sha(part)==item['lfs']['oid'],('Pinned blob mismatch',subject)
        assert not dest.exists();part.rename(dest)
        return {'subject':subject,'source_path':name,'file':dest.relative_to(root).as_posix(),'bytes':item['size'],'sha256':item['lfs']['oid'],'url':url,'status':'downloaded_hash_verified','resumed_from_bytes':offset}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(acquire,item):item['path'] for item in files}
        for future in concurrent.futures.as_completed(pending):
            name=pending[future]
            try:
                record=future.result();results.append(record)
                print('MMMU_VALIDATION_FILE_VERIFIED',record['subject'],record['bytes'],flush=True)
            except BaseException as error:
                record={'source_path':name,'error_type':type(error).__name__,'error':str(error)};errors.append(record)
                print('MMMU_VALIDATION_FILE_FAILED',name,type(error).__name__,flush=True)
            record['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            with events.open('ab') as stream:stream.write((json.dumps(record,sort_keys=True)+'\n').encode())
    data={'status':'mmmu_validation_sources_acquired' if len(results)==30 and not errors else 'mmmu_validation_acquisition_incomplete',
        'dataset':'MMMU/MMMU','revision':info['sha'],'api_sha256':sha(api),'tree_sha256':sha(tree),'method_sha256':sha(Path(__file__)),
        'files':sorted(results,key=lambda r:r['source_path']),'errors':errors,'gpu_work':False,'inference_requests':0,
        'scope':'Pinned public validation Parquet acquisition only. Dataset answers/questions not yet inspected for supported single-image/context coverage. No model output, quality score, timing or performance admission. Images remain local; no dataset payload is committed to the research branch.'}
    if report.exists():
        previous=json.loads(report.read_bytes());assert previous['dataset']==data['dataset'] and previous['revision']==data['revision']
        history=root/('validation-acquisition-prior-'+sha(report)[:16]+'.json');assert not history.exists();report.rename(history)
    report.write_text(json.dumps(data,indent=2),encoding='utf8')
    assert data['status']=='mmmu_validation_sources_acquired'
    print(json.dumps({'status':data['status'],'files':len(results),'bytes':sum(r['bytes'] for r in results),'gpu_work':False}))


if __name__=='__main__':main()
