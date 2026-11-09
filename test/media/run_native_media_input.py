# SPDX-License-Identifier: Apache-2.0
"""Native audio decode/resample, Slaney frontend and image preprocessing checks."""
import json,math,os,struct,sys,wave
sys.path.insert(0,os.getcwd())
import garnet,tensor
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True);path=os.path.join(root,'tone.wav')
rate=24000;n=2400
samples=[int(12000*math.sin(2*math.pi*440*i/rate)) for i in range(n)]
with wave.open(path,'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(struct.pack('<'+'h'*n,*samples))
media=garnet.media();audio=media.decode_audio(path,16000);assert list(audio.shape)==[1,2,1600],audio.shape
with open(path,'rb') as source:wave_bytes=source.read()
assert media.decode_audio(wave_bytes,16000).tolist()==audio.tolist()
assert media.decode_audio_range(wave_bytes,16000,20,100).tolist()==media.audio_segment(audio,20,100).tolist()
values=audio.tolist();assert max(abs(a-b) for a,b in zip(values[:1600],values[1600:]))<1e-6
assert sum(x*x for x in values)>1
energy=sum(x*x for x in values[:1600])/1600
assert abs(energy/((12000/32768)**2/2)-1)<.02,'mono duplication changed signal amplitude'
segment=media.audio_segment(audio,1550,100);assert list(segment.shape)==[1,2,100]
assert segment.tolist()[:50]==values[1550:1600] and segment.tolist()[50:100]==[0.0]*50
mel=media.audio_log_mel(audio,16000,1024,160,64);assert list(mel.shape)==[1,2,11,64]
assert all(math.isfinite(x) for x in mel.tolist())
# A silent input must map exactly to log(1e-5), independently of FFT/filter details.
silence=tensor.tensor([0.0]*3200,shape=[1,2,1600],dtype=tensor.float32)
assert max(abs(x-math.log(1e-5)) for x in media.audio_log_mel(silence,16000,1024,160,64).tolist())<1e-6
# Compare one tone frame to an independent direct-DFT magnitude + Slaney integral.
data=values[:1600];spectrum=[]
for k in range(513):
    re=0.0;im=0.0
    for j in range(1024):
        idx=abs(j-512);x=data[idx]*(.5-.5*math.cos(2*math.pi*j/1024));angle=2*math.pi*k*j/1024
        re+=x*math.cos(angle);im-=x*math.sin(angle)
    spectrum.append(math.sqrt(re*re+im*im))
def hz(m):return m*(200/3) if m<15 else 1000*math.exp((m-15)*math.log(6.4)/27)
top=15+math.log(8)/(math.log(6.4)/27);edges=[hz(top*i/65) for i in range(66)];expected=[]
for b in range(64):
    energy=0.0
    for k in range(513):
        freq=k*16000/1024;weight=max(0,min((freq-edges[b])/(edges[b+1]-edges[b]),(edges[b+2]-freq)/(edges[b+2]-edges[b+1])))*2/(edges[b+2]-edges[b])
        energy+=spectrum[k]*weight
    expected.append(math.log(max(energy,1e-5)))
assert max(abs(a-b) for a,b in zip(mel.tolist()[:64],expected))<2e-5
image=tensor.tensor([-.5]*12+[.25]*12+[.75]*12,shape=[1,3,3,4],dtype=tensor.float32)
resized=media.resize_image(image,8,8);assert list(resized.shape)==[1,3,8,8]
assert max(abs(a-b) for a,b in zip(resized.tolist(),[-.5]*64+[.25]*64+[.75]*64))<1e-6
for bad in (0,7999,192001):
    rejected=False
    try:media.decode_audio(path,bad)
    except Exception:rejected=True
    assert rejected
print('Native media input passed: stereo resampling, independent DFT/Slaney parity, silence, resizing, rejection',flush=True)
