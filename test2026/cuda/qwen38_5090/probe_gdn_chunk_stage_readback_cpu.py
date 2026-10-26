"""Embedded interpreter readback compatibility only; no Garnet/CUDA import or model."""
from pathlib import Path
import json

def main():
    out=Path('D:/CantorAI/5090-study-20261023/results/gdn-chunk-stage-readback-cpu');assert not out.exists();out.mkdir()
    (out/'a.f32').write_bytes(b'abcd');(out/'b.f32').write_bytes(b'efgh')
    row={'layer':0};entry={'parts':[{'file':'a.f32'},{'file':'b.f32'}]}
    def original(item):return b''.join((out[row['file']]).read_bytes() for row in item['parts'])
    def loop(item):
        values=[]
        for piece in item['parts']:
            values.append((out/piece['file']).read_bytes())
        return b''.join(values)
    def concatenate(item):
        result=b''
        for piece in item['parts']:result+=(out/piece['file']).read_bytes()
        return result
    error=None;old=None
    try:old=original(entry)
    except Exception as exc:error=repr(exc)
    result=loop(entry);other=concatenate(entry);assert result==other==b'abcdefgh'
    report={'status':'embedded_readback_explicit_loop_passed','original_error':error,'original_exact':old==b'abcdefgh' if error is None else None,'loop_exact':result==b'abcdefgh','concatenation_exact':other==b'abcdefgh','gpu_or_native_model_executed':False,'sdk_modified':False}
    (out/'report.json').write_bytes(json.dumps(report,indent=2).encode());print(json.dumps(report));return 0

if __name__=='__main__':raise SystemExit(main())
