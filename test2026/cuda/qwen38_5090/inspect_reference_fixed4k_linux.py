"""Read-only installed-source/checkpoint/PID inspection; no model or GPU imports."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path('/mnt/d/CantorAI/5090-study-20261023')
M=ROOT/'Garnet/test2026/cuda/qwen38_5090'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--engine',choices=('vllm','sglang'),required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output.resolve();assert out.is_relative_to((ROOT/'results').resolve()) and out.parent.is_dir() and not out.exists()
 pins_path=M/'evidence/reference4k-source-pins.json';pins=json.loads(pins_path.read_bytes());pid=int((out.parent/'server.pid').read_text().strip());assert pid>1
 proc=Path('/proc')/str(pid);cmdline=(proc/'cmdline').read_bytes();argv=cmdline.decode().strip('\0').split('\0');stat=(proc/'stat').read_bytes();starttime=stat.rsplit(b')',1)[1].split()[19]
 checkpoint=ROOT/'models/Qwen3.8-27B-NVFP4';port='18080' if args.engine=='vllm' else '18081';prefix='/opt/cantorai5090/'+args.engine+'/bin/'
 assert argv[0].startswith(prefix) and str(checkpoint) in argv and argv[argv.index('--port')+1]==port and argv[argv.index('--host')+1]=='127.0.0.1'
 source_root=out.parent/'installed-source';assert not source_root.exists();sources={}
 for path,expected in pins['linux_package_files'][args.engine].items():
  f=Path(path);site=Path('/opt/cantorai5090')/args.engine/'lib/python3.12/site-packages';relative=f.relative_to(site);raw=f.read_bytes();assert sha(raw)==expected,('Installed source drift',path)
  target=source_root/relative;assert target.resolve().is_relative_to(source_root.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);sources[path]={'sha256':expected,'archived_copy':target.relative_to(out.parent).as_posix()}
 checkpoint_hashes={}
 for name,expected in pins['checkpoint_files'].items():
  raw=(checkpoint/name).read_bytes();assert sha(raw)==expected,('Checkpoint metadata drift',name);checkpoint_hashes[name]=expected
 assert (proc/'cmdline').read_bytes()==cmdline and (proc/'stat').read_bytes().rsplit(b')',1)[1].split()[19]==starttime
 result={'status':'reference_fixed4k_installed_sources_and_owned_parent_verified','engine':args.engine,'pid':pid,'linux_proc_starttime_ticks':int(starttime),'cmdline':argv,'cmdline_sha256':sha(cmdline),'source_pins_sha256':sha(pins_path.read_bytes()),'installed_sources':sources,'checkpoint_files':checkpoint_hashes,'checkpoint_revision':pins['checkpoint_revision'],'scope':'Exact named installed-source/checkpoint-metadata hashes and owned parent command/starttime before quality/timing. Not every installed binary or checkpoint weight rehashed, no quantized-kernel/tuning/quality/speed admission.'};out.write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'engine':args.engine,'pid':pid,'sources':len(sources),'checkpoint_files':len(checkpoint_hashes)}));return 0
if __name__=='__main__':raise SystemExit(main())
