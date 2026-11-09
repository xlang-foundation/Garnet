# SPDX-License-Identifier: Apache-2.0
"""Validate native generated-PCM/AAC muxing with independent FFmpeg decoding."""
import json,math,os,socket,struct,subprocess,sys,threading
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
import garnet,tensor
root=os.path.abspath(sys.argv[1]);probe=os.path.abspath(sys.argv[2]);ffmpeg=os.path.abspath(sys.argv[3])
os.makedirs(root,exist_ok=True)
capabilities=json.loads(garnet.media().require_provider('ffmpeg'))
assert 'aac' in capabilities.get('audio_codecs',[]),capabilities
rate=24000;samples=9000
pcm=[0.2*math.sin(2*math.pi*frequency*i/rate) for frequency in (440,880) for i in range(samples)]
path=os.path.join(root,'joint-output.mp4')
writer=garnet.video_writer(json.dumps({'path':path,'width':256,'height':256,'fps':24,'allow_hardware':False,
    'audio_sample_rate':rate,'audio_channels':2,'queue_capacity':2}))
invalid=tensor.tensor([float('nan')]*samples*2,shape=[1,2,samples],dtype=tensor.float32)
rejected=False
try: writer.write_audio(invalid,rate)
except Exception: rejected=True
assert rejected,'nonfinite audio was accepted'
writer.write_audio(tensor.tensor(pcm,shape=[1,2,samples],dtype=tensor.float32),rate)
writer.write_chunk(tensor.tensor(shape=[1,3,9,256,256],dtype=tensor.float32))
result=json.loads(writer.finish());assert result['audio_samples']==samples and result['frames_consumed']==9,result
output=subprocess.run([probe,'-v','error','-show_entries','stream=codec_type,codec_name,sample_rate,channels,duration,nb_frames','-of','json',path],capture_output=True,text=True)
assert output.returncode==0,output.stderr
streams=json.loads(output.stdout)['streams'];audio=[s for s in streams if s['codec_type']=='audio'][0]
video=[s for s in streams if s['codec_type']=='video'][0]
assert audio['codec_name']=='aac' and audio['sample_rate']==str(rate) and audio['channels']==2,audio
assert abs(float(audio['duration'])-samples/rate)<0.002,audio
assert video['nb_frames']=='9' and abs(float(video['duration'])-samples/rate)<0.002,video
decoded=os.path.join(root,'decoded.f32')
output=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',path,'-map','0:a:0','-f','f32le',decoded],capture_output=True,text=True)
assert output.returncode==0,output.stderr
with open(decoded,'rb') as source: raw=source.read()
values=struct.unpack('<'+'f'*(len(raw)//4),raw)
assert len(values)>=samples*2
for c in range(2):
    actual=[values[i*2+c] for i in range(512,samples-512)]
    expected=pcm[c*samples+512:(c+1)*samples-512]
    correlation=sum(a*b for a,b in zip(actual,expected))/math.sqrt(sum(a*a for a in actual)*sum(b*b for b in expected))
    assert correlation>0.98,(c,correlation)
print('Native video + generated stereo audio passed: AAC, timestamps, duration, channel identity, signal correlation and nonfinite rejection',flush=True)

# Exercise the same non-seekable loopback transport used by Galaxy workers.
with socket.socket() as listener:
    listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(10)
    target='tcp://127.0.0.1:'+str(listener.getsockname()[1])
    stream_path=os.path.join(root,'streamed-output.mp4')
    stream_errors=[]
    def capture():
        try:
            conn,_=listener.accept()
            with conn,open(stream_path,'wb') as sink:
                conn.settimeout(10)
                while True:
                    chunk=conn.recv(65536)
                    if not chunk:break
                    sink.write(chunk)
        except Exception as error:stream_errors.append(error)
    receiver=threading.Thread(target=capture);receiver.start()
    stream_writer=None
    try:
        stream_writer=garnet.video_assembler(json.dumps({'path':target,'width':256,'height':256,'fps':24,'frames':9,
            'allow_hardware':False,'audio_sample_rate':rate,'audio_channels':2,'queue_capacity':2,'tile_spool_memory_mib':8,
            'tiles':[{'x':x,'y':0,'width':128,'height':256,'feather':[0,0,0,0]} for x in (0,128)]}))
        stream_writer.write_audio(tensor.tensor(pcm,shape=[1,2,samples],dtype=tensor.float32),rate)
        tile=tensor.tensor(shape=[1,3,9,256,128],dtype=tensor.float32)
        stream_writer.add_tile(0,tile);stream_writer.add_tile(1,tile)
        result=json.loads(stream_writer.finish())
        assert result['audio_samples']==samples and result['frames_consumed']==9,result
        assert result['spool_bytes']==0 and 0<result['memory_spool_peak_bytes']<=8*1024*1024,result
    finally:
        if stream_writer is not None:stream_writer.abort()
        receiver.join()
    assert not stream_errors,stream_errors
result=subprocess.run([probe,'-v','error','-count_frames','-show_entries',
    'stream=codec_type,codec_name,nb_read_frames,start_time,duration','-of','json',stream_path],capture_output=True,text=True)
assert result.returncode==0,result.stderr
streams={item['codec_type']:item for item in json.loads(result.stdout)['streams']}
assert streams['video']['codec_name']=='h264' and streams['video']['nb_read_frames']=='9',streams
assert streams['audio']['codec_name']=='aac',streams
assert abs(float(streams['video']['start_time'])-float(streams['audio']['start_time']))<0.020,streams
assert abs(float(streams['video']['duration'])-samples/rate)<0.002,streams
stream_pcm=os.path.join(root,'streamed-decoded.f32')
result=subprocess.run([ffmpeg,'-v','error','-y','-i',stream_path,'-map','0:a:0','-f','f32le',stream_pcm],capture_output=True,text=True)
assert result.returncode==0,result.stderr
with open(stream_pcm,'rb') as source:stream_raw=source.read()
assert stream_raw[:samples*8]==raw[:samples*8],'stream transport shifted or changed decoded audio'
print('Native fragmented MP4 loopback transport passed',flush=True)

# The noncausal LTX VAE submits a complete planar tile. Check its native
# rearrangement into 1+4+4 frame chunks independently with lossless PNG exports.
tile_path=os.path.join(root,'joint-tile.mp4')
assembler=garnet.video_assembler(json.dumps({'path':tile_path,'width':256,'height':256,'fps':24,'frames':9,
    'allow_hardware':False,'audio_sample_rate':rate,'audio_channels':2,'save_frames':True,'queue_capacity':2,
    'spool_directory':os.path.join(root,'spool'),'tiles':[{'x':0,'y':0,'width':256,'height':256,'feather':[0,0,0,0]}]}))
assembler.write_audio(tensor.tensor(pcm,shape=[1,2,samples],dtype=tensor.float32),rate)
# Float32 striped values exercise clipping, ties and nearby rounding cases.
palette=[struct.unpack('<f',struct.pack('<f',value))[0] for value in [-2.,-1.,-1/255,0.,1/255,1.,2.]]
pixels=[palette[(x+f+c)%len(palette)] for c in range(3) for f in range(9) for y in range(256) for x in range(256)]
assembler.add_tile(0,tensor.tensor(pixels,shape=[1,3,9,256,256],dtype=tensor.float32))
result=json.loads(assembler.finish());assert result['frames_consumed']==9 and result['audio_samples']==samples,result
assert result['queue_peak_frames']<=2 and result['assembly_peak_pixel_bytes']<=4*4*256*256*4,result
assert result['assembly_direct_full_canvas'] and result['blend_workspace_peak_bytes']==0,result
assert result['assembly_peak_pixel_bytes']==4*256*256*3,result
for frame in range(9):
    source=os.path.join(tile_path+'.frames','frame-'+str(frame).zfill(6)+'.png')
    decoded=os.path.join(root,'tile-frame-'+str(frame)+'.rgb')
    output=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',source,'-f','rawvideo','-pix_fmt','rgb24',decoded],capture_output=True,text=True)
    assert output.returncode==0,output.stderr
    with open(decoded,'rb') as f:rgb=f.read()
    expected=bytes([round(max(0,min(255,(palette[(x+frame+c)%len(palette)]+1)*127.5))) for x in range(256) for c in range(3)])*256
    assert rgb==expected,(frame,rgb[:3],expected[:3])

# Deferred audio uses the same PCM and timestamps after the last video tile.
late_path=os.path.join(root,'joint-tile-late-audio.mp4')
late=garnet.video_assembler(json.dumps({'path':late_path,'width':256,'height':256,'fps':24,'frames':9,
    'allow_hardware':False,'audio_sample_rate':rate,'audio_channels':2,'audio_at_finish':True,
    'spool_directory':os.path.join(root,'spool'),'tiles':[{'x':0,'y':0,'width':256,'height':256,'feather':[0,0,0,0]}]}))
late.add_tile(0,tensor.tensor(pixels,shape=[1,3,9,256,256],dtype=tensor.float32))
result=json.loads(late.finish_with_audio(tensor.tensor(pcm,shape=[1,2,samples],dtype=tensor.float32),rate))
assert result['frames_consumed']==9 and result['audio_samples']==samples,result
for source,track,expected_path in ((late_path,'0:a:0',path),(late_path,'0:v:0',tile_path)):
    actual=subprocess.run([ffmpeg,'-v','error','-i',source,'-map',track,'-f','framemd5','-'],capture_output=True,text=True)
    expected=subprocess.run([ffmpeg,'-v','error','-i',expected_path,'-map',track,'-f','framemd5','-'],capture_output=True,text=True)
    assert actual.returncode==0 and expected.returncode==0 and actual.stdout==expected.stdout,(track,actual.stderr,expected.stderr)

# Failure in a parallel conversion must propagate and join workers before the
# source tensor and assembler are destroyed.
invalid_path=os.path.join(root,'invalid-tile.mp4')
invalid=garnet.video_assembler(json.dumps({'path':invalid_path,'width':64,'height':64,'fps':24,'frames':5,
    'allow_hardware':False,'spool_directory':os.path.join(root,'spool'),
    'tiles':[{'x':0,'y':0,'width':64,'height':64,'feather':[0,0,0,0]}]}))
bad=[0.]*(3*5*64*64);bad[-1]=float('nan');rejected=False
try:invalid.add_tile(0,tensor.tensor(bad,shape=[1,3,5,64,64],dtype=tensor.float32))
except Exception:rejected=True
finally:invalid.abort()
assert rejected,'nonfinite pixel in a parallel tile conversion was hidden'
print('Native complete-tile assembly passed: planar channel/frame order, bounded queue and joint audio',flush=True)

# The two windows own frames 0..4 and 5..8. Their PCM boundary crosses
# an AAC frame and must not introduce padding between the two submissions.
movie_path=os.path.join(root,'joint-streaming-movie.mp4')
movie=garnet.video_writer(json.dumps({'path':movie_path,'width':256,'height':256,'fps':24,
    'allow_hardware':False,'audio_sample_rate':rate,'audio_channels':2,'streaming_audio':True,'queue_capacity':2}))
window_options={'width':256,'height':256,'fps':24,'frames':5,'spool_directory':os.path.join(root,'spool'),
    'tiles':[{'x':0,'y':0,'width':256,'height':256,'feather':[0,0,0,0]}]}
plane=256*256
for first,skip,offset,count in ((0,0,0,5000),(4,1,5000,4000)):
    part=[value for channel in range(2) for value in pcm[channel*samples+offset:channel*samples+offset+count]]
    movie.write_audio_chunk(tensor.tensor(part,shape=[1,2,count],dtype=tensor.float32),rate)
    movie.begin_window(json.dumps(window_options),skip)
    tile=[value for channel in range(3)
        for value in pixels[(channel*9+first)*plane:(channel*9+first+5)*plane]]
    movie.add_window_tile(0,tensor.tensor(tile,shape=[1,3,5,256,256],dtype=tensor.float32))
    movie.finish_window()
result=json.loads(movie.finish())
assert result['frames_consumed']==9 and result['audio_samples']==samples,result
for track in ('0:a:0','0:v:0'):
    actual=subprocess.run([ffmpeg,'-v','error','-i',movie_path,'-map',track,'-f','framemd5','-'],capture_output=True,text=True)
    expected=subprocess.run([ffmpeg,'-v','error','-i',tile_path,'-map',track,'-f','framemd5','-'],capture_output=True,text=True)
    assert actual.returncode==0 and expected.returncode==0 and actual.stdout==expected.stdout,(track,actual.stderr,expected.stderr)
media=garnet.media()
full=media.decode_audio(movie_path,48000)
selected=media.decode_audio_range(movie_path,48000,2345,1001)
assert selected.tolist()==media.audio_segment(full,2345,1001).tolist()
rejected=False
try:media.decode_audio_range(movie_path,48000,full.shape[-1],1)
except Exception:rejected=True
assert rejected,'a short input audio range was accepted'
print('Streaming movie and bounded audio ranges passed with exact decoded frames and timestamps',flush=True)
