"""Read and archive actual installed context-limit sources; no framework imports."""
from pathlib import Path
import argparse,hashlib,json
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True);args=a.parse_args();target=args.output.resolve();root=Path('/mnt/d/CantorAI/5090-study-20261023');assert target.is_relative_to(root/'results') and target.parent.is_dir() and not target.exists()
 pins_path=Path(__file__).with_name('evidence')/'sglang-context-limit-source-pins.json';pins=json.loads(pins_path.read_bytes());package=Path(pins['package_root']);files={}
 for name,digest in pins['files'].items():
  path=package/name;raw=path.read_bytes();assert sha(raw)==digest,('Installed context source drift',name)
  relative='installed-context-source/'+name;copy=target.parent/relative;assert not copy.exists();copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(raw);files[name]={'sha256':digest,'archived_copy':relative}
 result={'status':'installed_context_limit_sources_verified','pins_sha256':sha(pins_path.read_bytes()),'files':files,'configured_engine_context_capacity':4098,'actual_workload_context_capacity':4096,'source_modified':False};target.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({'status':result['status'],'files':len(files)}));return 0
if __name__=='__main__':raise SystemExit(main())
