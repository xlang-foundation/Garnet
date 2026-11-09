# SPDX-License-Identifier: Apache-2.0
"""Real native provider tests. Run with XLang3, not CPython."""
import json
import math
import os
import struct
import subprocess
import sys
import time
import zlib
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
import garnet
import tensor
from xModel.generation.image_io import read_png_rgb,png_rgb,load_rgb_tensor
root=os.path.abspath(sys.argv[1]);probe=os.path.abspath(sys.argv[2]);os.makedirs(root,exist_ok=True)
media=garnet.media()
assert 'png' in media.require_provider('image')
assert 'h264' in media.require_provider('ffmpeg')

# Independently encoded scanlines exercise every PNG filter in the native decoder.
def chunk(kind,data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
pixels=bytes([255,0,0,0,255,0,0,0,255,255,255,255])
header=png_rgb(2,2,pixels)[:33]
for filtering,row in enumerate([[0,0,255,255,255,255],[0,0,255,255,255,0],[1,0,255,255,0,255],
    [129,0,255,255,128,128],[1,0,255,255,0,0]]):
    path=os.path.join(root,'filter-'+str(filtering)+'.png')
    with open(path,'wb') as target:
        target.write(header+chunk(b'IDAT',zlib.compress(b'\x00'+pixels[:6]+bytes([filtering])+bytes(row)))+chunk(b'IEND',b''))
    decoded=load_rgb_tensor(path)
    assert list(decoded.shape)==[1,3,2,2] and decoded.tolist()==[1,-1,-1,1,-1,1,-1,1,-1,-1,1,1]
bad_path=os.path.join(root,'invalid.jpg')
with open(bad_path,'wb') as target: target.write(b'invalid jpeg')
rejected=False
try: media.decode_image(bad_path)
except Exception: rejected=True
assert rejected,'invalid image was accepted'

# Reject malformed admission settings before opening a codec or allocating frames.
for field,value in (('width',256.5),('height',-1),('fps',24.5),('queue_capacity',17),('b_frames',1000),
    ('snapshot_workers',0),('snapshot_workers',17),('snapshot_workers',2.5)):
    options={'path':os.path.join(root,'invalid.mp4'),'width':256,'height':256,'fps':24,'allow_hardware':False}
    options[field]=value
    rejected=False
    try: garnet.video_writer(json.dumps(options))
    except Exception: rejected=True
    assert rejected,(field,value)
def check_video(path,frames,width,height):
    process=subprocess.run([probe,'-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,avg_frame_rate','-of','json',path],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    stream=json.loads(process.stdout)['streams'][0]
    assert stream['codec_name']=='h264' and stream['nb_frames']==str(frames) and stream['width']==width and stream['height']==height and stream['avg_frame_rate']=='24/1',stream
    return stream
image=tensor.tensor([-0.6]*(256*256)+[0.0]*(256*256)+[0.6]*(256*256),shape=[1,3,256,256],dtype=tensor.float32)
expected=garnet.tensor_pixels().rgb8(image,0)
png=os.path.join(root,'native.png');media.encode_image(image,png,'{"format":"png"}')
assert bytes(read_png_rgb(png)[2])==expected
decoded=media.decode_image(png);assert list(decoded.shape)==[1,3,256,256]
png_bytes=media.encode_image(image,'','{"format":"png"}')
with open(png,'rb') as source:assert png_bytes==source.read()
assert media.decode_image(png_bytes).tolist()==decoded.tolist()
jpeg=os.path.join(root,'cpu.jpg');media.encode_image(image,jpeg,'{"format":"jpeg","quality":95}')
assert max(abs(a-b) for a,b in zip(image.tolist(),media.decode_image(jpeg).tolist()))<0.03
with open(jpeg,'rb') as source:assert media.decode_image(source.read()).tolist()==media.decode_image(jpeg).tolist()
if os.environ.get('GARNET_TEST_MEDIA_CUDA')=='1':
    planes=garnet.tensor_to_gpu(tensor.tensor(list(range(120)),shape=[1,2,3,4,5],dtype=tensor.float32))
    io=garnet.cuda_tensor_io()
    cropped=io.crop_planes(planes,1,2,2,3)
    assert list(cropped.shape)==[1,2,3,2,3]
    assert garnet.tensor_to_cpu(cropped).tolist()==[20*plane+5*y+x for plane in range(6) for y in (1,2) for x in (2,3,4)]
    rejected=False
    try:io.crop_planes(planes,3,2,2,3)
    except Exception:rejected=True
    assert rejected,'out-of-bounds plane crop was accepted'
    gpu=garnet.tensor_to_gpu(image)
    media.encode_image(gpu,os.path.join(root,'gpu.jpg'),'{"format":"jpeg","quality":95,"provider":"nvjpeg"}')
    assert max(abs(a-b) for a,b in zip(image.tolist(),media.decode_image(os.path.join(root,'gpu.jpg')).tolist()))<0.03
    bad=garnet.tensor_to_gpu(tensor.tensor([float('nan'),0,0],shape=[1,3,1,1],dtype=tensor.float32))
    rejected=False
    try: media.encode_image(bad,os.path.join(root,'invalid.jpg'),'{"format":"jpeg","provider":"nvjpeg"}')
    except Exception: rejected=True
    assert rejected,'nonfinite GPU pixels were accepted'
    print('Native CUDA nvJPEG encode/decode and rejection checks passed',flush=True)
video=tensor.tensor([0.6 if c==f%3 else -0.6 for c in range(3) for f in range(9) for _ in range(256*256)],shape=[1,3,9,256,256],dtype=tensor.float32)
encoders=['h264_mf' if os.name=='nt' else 'libx264']
if os.environ.get('GARNET_TEST_MEDIA_CUDA')=='1': encoders.append('h264_nvenc')
for encoder in encoders:
    reference=[]
    for workers in (1,4):
        path=os.path.join(root,encoder+('' if workers==1 else '-parallel')+'.mp4')
        writer=garnet.video_writer(json.dumps({'path':path,'width':256,'height':256,'fps':24,'encoder':encoder,'device_id':0,'queue_capacity':2,'b_frames':2,'save_frames':True,'snapshot_workers':workers}))
        writer.write_chunk(video);result=json.loads(writer.finish());assert result['frames']==9 and result['frames_consumed']==9 and result['queue_peak_frames']<=2,result
        assert result['snapshot_workers']==workers and result['queue_max_pixel_bytes']==(4+(workers if workers>1 else 0))*256*256*3,result
        assert json.loads(writer.finish())==result
        assert result['encoder']['hardware']==(encoder=='h264_nvenc'),result
        check_video(path,9,256,256)
        for f in range(9):
            frame=os.path.join(path+'.frames','frame-'+str(f).zfill(6)+'.png')
            assert bytes(read_png_rgb(frame)[2])==garnet.tensor_pixels().rgb8(video,f)
            with open(frame,'rb') as source:payload=source.read()
            if workers==1:reference.append(payload)
            else:assert payload==reference[f],'Parallel snapshot bytes changed'
        print('Native encoder, timestamps, snapshots and end-of-stream drain passed',encoder,workers,flush=True)
# A snapshot task failure must reach Finish and join all outstanding tasks.
bad_video=os.path.join(root,'snapshot-failure.mp4')
os.makedirs(os.path.join(bad_video+'.frames','frame-000000.png'),exist_ok=True)
writer=garnet.video_writer(json.dumps({'path':bad_video,'width':256,'height':256,'fps':24,'encoder':encoders[0],'save_frames':True,'snapshot_workers':4}))
rejected=False
try:writer.write_chunk(video);writer.finish()
except Exception:rejected=True
assert rejected,'Parallel snapshot failure was hidden'
writer=None
# Compare disk-backed assembly with the original independent full-canvas path.
width,height,frames=512,256,9
canvas=tensor.tensor(shape=[1,3,frames,height,width],dtype=tensor.float32)
coverage=tensor.tensor(shape=[frames,height,width],dtype=tensor.float32)
tiles=[{'x':x,'y':0,'width':256,'height':256,'feather':[0,128 if x else 0,0,128 if x<256 else 0]} for x in (0,128,256)]
spool=os.path.join(root,'spool');os.makedirs(spool,exist_ok=True)
with open(os.path.join(spool,'keep.txt'),'w') as output: output.write('user-owned sentinel')
output=os.path.join(root,'assembled.mp4')
assembly_options={'path':output,'width':width,'height':height,'frames':frames,'fps':24,
    'encoder':encoders[0],'queue_capacity':2,'save_frames':True,'tiles':tiles,'spool_directory':spool}
assembler=garnet.video_assembler(json.dumps(assembly_options))
mixed_output=os.path.join(root,'assembled-mixed-spool.mp4')
# One 256x256 RGB FP32 frame fits in 1 MiB. Four frames spill to disk.
mixed_options=dict(assembly_options,path=mixed_output,tile_spool_memory_mib=1)
mixed=garnet.video_assembler(json.dumps(mixed_options))
processor=garnet.tensor_pixels()
for index,tile in enumerate(tiles):
    values=[-0.7+c*0.25+f*0.02+(tile['x']+x)*0.0003+y*0.0001 for c in range(3) for f in range(frames) for y in range(height) for x in range(256)]
    whole=tensor.tensor(values,shape=[1,3,frames,height,256],dtype=tensor.float32)
    processor.blend_into(whole,canvas,coverage,0,0,tile['x'],tile['feather'])
    for first,count in ((0,1),(1,4),(5,4)):
        data=[values[((c*frames+f)*height+y)*256+x] for c in range(3) for f in range(first,first+count) for y in range(height) for x in range(256)]
        chunk=tensor.tensor(data,shape=[1,3,count,height,256],dtype=tensor.float32)
        assembler.add_tile_chunk(index,first,chunk)
        mixed.add_tile_chunk(index,first,chunk)
        if index==len(tiles)-1: time.sleep(0.1)
result=json.loads(assembler.finish());mixed_result=json.loads(mixed.finish());processor.normalize(canvas,coverage)
assert result['assembly_peak_pixel_bytes']==(3*4+1)*height*width*4,result
assert result['frames_consumed_before_final_chunk']>0,result
assert result['queue_peak_frames']<=2 and os.listdir(spool)==['keep.txt'],result
assert mixed_result['memory_spool_peak_bytes']==3*256*256*4,mixed_result
assert mixed_result['spool_bytes']+mixed_result['memory_spool_peak_bytes']==result['spool_bytes'],mixed_result
for f in range(frames):
    expected=processor.rgb8(canvas,f)
    for path in (output,mixed_output):
        assert bytes(read_png_rgb(os.path.join(path+'.frames','frame-'+str(f).zfill(6)+'.png'))[2])==expected,(path,f)
check_video(output,frames,width,height)
check_video(mixed_output,frames,width,height)
with open(os.path.join(root,'assembly-result.json'),'w') as out: json.dump(result,out,indent=2)
# Aborting a partial tile spool releases owned temporary files only.
aborted=garnet.video_assembler(json.dumps({'path':os.path.join(root,'aborted.mp4'),'width':width,'height':height,'frames':frames,
    'fps':24,'encoder':encoders[0],'tiles':tiles,'spool_directory':spool}))
aborted.add_tile_chunk(0,0,tensor.tensor(shape=[1,3,1,height,256],dtype=tensor.float32))
assert len(os.listdir(spool))==2
aborted.abort();assert os.listdir(spool)==['keep.txt']
# A fresh process imports and uses Garnet with the optional provider directory absent.
environment=dict(os.environ);environment['GARNET_MEDIA_PLUGIN_DIR']=os.path.join(root,'not-installed')
code='import garnet,tensor; t=tensor.tensor([1,2],dtype=tensor.float32); assert garnet.tensor_views().reshape(t,[2]).tolist()==[1,2]; failed=False\ntry: garnet.media().require_provider("ffmpeg")\nexcept Exception: failed=True\nassert failed;print("optional-provider-missing-passed")'
child=subprocess.run([sys.executable,'-c',code],env=environment,capture_output=True,text=True)
assert child.returncode==0,(child.stdout,child.stderr)
print('Native media providers passed: image pixels, codec output, bounded queue, exact tile assembly, cleanup and optional loading',flush=True)
