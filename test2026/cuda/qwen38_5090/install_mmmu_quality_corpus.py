"""Install/reuse the one frozen CPU corpus; never replace differing existing bytes."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
ARCHIVE_SHA256='ac3d9dd53a509f1884fe80c15b7c2737666013c7bd6a40e844672ecd2c00f6ae'
MANIFEST_SHA256='108d017c7db5be56c58f490c2a985dfbcde2e6ded0860a97e797cfac9cea1589'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def install(archive,target):
 archive=Path(archive);target=Path(target);assert sha(archive.read_bytes())==ARCHIVE_SHA256
 assert target.parent.is_dir() and not target.is_symlink()
 created=[];reused=0
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  raw=z.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;manifest=json.loads(raw);expected=manifest['files']
  assert set(z.namelist())==set(expected)|{'manifest.json'} and manifest['requests']==805
  for n in z.namelist():
   relative=PurePosixPath(n);assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in n and ':' not in n
   payload=z.read(n)
   if n!='manifest.json':assert sha(payload)==expected[n]['sha256'] and len(payload)==expected[n]['bytes']
  target.mkdir(exist_ok=True);resolved=target.resolve(strict=True)
  assert not(getattr(target.lstat(),'st_file_attributes',0)&0x400),'Corpus root must not be a reparse point'
  actual={f.relative_to(target).as_posix() for f in target.rglob('*') if f.is_file()};assert actual<=set(z.namelist()),'Preserve unknown corpus files'
  for n in actual:
   f=target/n;assert f.resolve().is_relative_to(resolved) and not f.is_symlink() and f.read_bytes()==z.read(n),('Preserve differing corpus bytes',n)
  for n in z.namelist():
   f=target/n;assert f.resolve().is_relative_to(resolved);payload=z.read(n)
   if f.exists():assert f.is_file() and not f.is_symlink() and f.read_bytes()==payload,('Preserve differing corpus bytes',n);reused+=1
   else:
    f.parent.mkdir(parents=True,exist_ok=True);assert f.parent.resolve().is_relative_to(resolved)
    with f.open('xb') as output:output.write(payload)
    created.append(n)
 return {'status':'frozen_mmmu_corpus_installed_or_reused','archive_sha256':ARCHIVE_SHA256,'manifest_sha256':MANIFEST_SHA256,'files_created':len(created),'files_reused':reused,'scope':'One fixed805request CPU corpus; no HTTP, model execution, accuracy or benchmark.'}
