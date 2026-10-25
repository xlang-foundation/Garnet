"""Collect the frozen805 adapted screening requests against one already-ready server.

This never starts a server. Establish exclusive GPU ownership and native/model
correctness gates with the study controller before invoking this client.
"""
import argparse,json
from pathlib import Path
from .mmmu_http_quality import load_corpus,collect,MODELS

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--engine',choices=('native','vllm','sglang'),required=True);p.add_argument('--model-alias',choices=MODELS,default=MODELS[0]);p.add_argument('--url',required=True)
 p.add_argument('--server-identity',type=Path,required=True);p.add_argument('--timeout',type=float,default=180);a=p.parse_args()
 _,rows,gold=load_corpus(a.corpus);identity=json.loads(a.server_identity.read_bytes())
 result=collect(a.corpus,rows,gold,a.output,a.engine,a.model_alias,a.url,identity,timeout=a.timeout)
 print(json.dumps({'status':result['status'],'requests':result['completed_requests'],'server_identity_attested_by_client':False,'runtime_image_preprocessing_verified':False,'quality_promotion_admitted':False,'performance_admitted':False,'scope':'Complete collection still needs independent replay; adapted805 low-resolution screening, not official900 MMMU or timing.'}))
 return 0
if __name__=='__main__':raise SystemExit(main())
