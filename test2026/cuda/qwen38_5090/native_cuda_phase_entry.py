"""Diagnostic entry log before native imports; original model operations retained.

No model/GPU import occurs when importing this module. The explicit entry point
runs only the existing qualified logged worker, preserving its globals/argv/IPC.
"""
from pathlib import Path
import hashlib,json,os,sys,time,traceback

HOOK='_garnet_entry_mark'
IMPORTS=(
    ('import garnet','garnet_import'),
    ('from xModel.qwen38.vl_27b.model import create_cuda_bf16_low_batch_study_decoder,create_cuda_vision,prepare_cuda_image','model_import'),
)


def instrument(source):
    if HOOK in source:raise ValueError('Preserve original uninstrumented worker')
    for statement,label in IMPORTS:
        needle='\n'+statement+'\n'
        if source.count(needle)!=1:raise ValueError('Exact qualified native import required: '+label)
        source=source.replace(needle,'\n'+HOOK+'('+repr('before_'+label)+')\n'+statement+'\n'+HOOK+'('+repr(label+'_returned')+')\n')
    return source


class EntryLog:
    def __init__(self,path,mode):
        self.path=Path(path);self.ordinal=0
        if mode not in ('ordinary','mtp'):raise ValueError('Explicit diagnostic mode required')
        self.mode=mode
        with self.path.open('x',encoding='utf8') as stream:
            stream.write(json.dumps({'ordinal':0,'stage':'python_entry_started','mode':mode,'pid':os.getpid()})+'\n')
            stream.flush();os.fsync(stream.fileno())
        self.ordinal=1

    def mark(self,stage,**extra):
        row={'ordinal':self.ordinal,'stage':stage,'mode':self.mode,'monotonic_seconds':time.monotonic(),**extra}
        with self.path.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        self.ordinal+=1


def execute(source,source_path,retained_path,log):
    raw=source.encode();candidate=instrument(source);data=candidate.encode()
    with Path(retained_path).open('xb') as stream:stream.write(data)
    log.mark('instrumented_source_retained',original_sha256=hashlib.sha256(raw).hexdigest(),
             instrumented_sha256=hashlib.sha256(data).hexdigest())
    namespace={'__name__':'__main__','__file__':str(source_path),HOOK:log.mark}
    try:
        exec(compile(candidate,str(retained_path),'exec'),namespace)
    except BaseException as error:
        log.mark('worker_raised',exception={'type':type(error).__name__,'repr':repr(error),
                 'traceback':''.join(traceback.format_exception(type(error),error,error.__traceback__))})
        raise
    log.mark('worker_execution_returned')


def main():
    if len(sys.argv)!=4:raise ValueError('Explicit root, mode and diagnostic trace path required')
    root=Path(sys.argv[1]).resolve();mode=sys.argv[2];trace=Path(sys.argv[3]).resolve()
    if root!=Path('D:/CantorAI/5090-study-20261023').resolve() or not trace.is_relative_to(root/'results'):
        raise ValueError('Fixed owned study results path required')
    if not trace.parent.is_dir():raise FileNotFoundError('Owned diagnostic output directory required')
    source_path=Path(__file__).with_name('native_cuda_phase_logged_worker.py')
    log=EntryLog(trace.with_suffix('.entry.jsonl'),mode)
    execute(source_path.read_bytes().decode(),source_path,trace.with_suffix('.entry-worker.py.txt'),log)
    return 0


if __name__=='__main__':raise SystemExit(main())
