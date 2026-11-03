"""Native XLang3 fixed-grid preprocessing without loading a model checkpoint."""
import os,sys,json
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
image,output=sys.argv[1:3]
io=garnet.cuda_tensor_io();checks=[]
try:
    for gh,gw in [(12,20),(20,12),(16,16)]:
        tensor,h,w=io.prepare_image_grid(image,gh,gw,65536)
        assert [h,w]==[gh,gw]
        assert len(io.finite_f32_bytes(tensor))==gh*gw*1536*4
        checks.append({'grid':[h,w],'bytes':gh*gw*1536*4})
    for gh,gw in [(11,20),(20,20),(0,20),(2147483647,2147483647)]:
        try:io.prepare_image_grid(image,gh,gw,65536)
        except Exception:pass
        else:raise AssertionError('Invalid grid admitted')
    Path(output).write_text(json.dumps({'passed':True,'runtime':sys.implementation.name,'checks':checks,'invalid_grids_rejected':4},indent=2))
    print('CUDA_IMAGE_GRID_PASS',len(checks),flush=True)
finally:io.close()
