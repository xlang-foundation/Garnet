# SPDX-License-Identifier: Apache-2.0
"""XLang3 checks for schedules, seeded noise, PNG layout, and shared views."""
import math
import os
import struct
import sys
import zlib
sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","..")))
import garnet
import tensor
from xModel.generation.flow import euler_sigmas,gaussian_noise
from xModel.generation.image_io import png_rgb,read_png_rgb
from xModel.flux2.klein_4b.latents import rope_tables

assert euler_sigmas(4)==[1,0.75,0.5,0.25,0]
assert euler_sigmas(2,shift=3)==[1,0.75,0]
assert euler_sigmas(4,mu=0)==euler_sigmas(4)
for mu in (-1.5,0,2.0):
    values=euler_sigmas(8,mu=mu)
    assert values[0]==1 and values[-1]==0 and all(a>b for a,b in zip(values,values[1:]))
noise=gaussian_noise([2,5000],123).tolist()
assert noise==gaussian_noise([2,5000],123).tolist()
assert noise!=gaussian_noise([2,5000],124).tolist()
assert abs(sum(noise)/len(noise))<0.05
assert abs(sum(x*x for x in noise)/len(noise)-1)<0.08

# Validate the file independently by parsing chunks and reconstructing scans.
pixels=bytes([255,0,0,0,255,0,0,0,255,255,255,255])
encoded=png_rgb(2,2,pixels)
assert encoded[:8]==b"\x89PNG\r\n\x1a\n"
offset=8
data=b""
while offset<len(encoded):
    size=struct.unpack(">I",encoded[offset:offset+4])[0]
    kind=encoded[offset+4:offset+8]
    payload=encoded[offset+8:offset+8+size]
    crc=struct.unpack(">I",encoded[offset+8+size:offset+12+size])[0]
    assert zlib.crc32(kind+payload)&0xffffffff==crc
    if kind==b"IHDR": assert struct.unpack(">IIBBBBB",payload)==(2,2,8,2,0,0,0)
    if kind==b"IDAT": data+=payload
    offset+=12+size
assert zlib.decompress(data)==b"\x00"+pixels[:6]+b"\x00"+pixels[6:]

# Fixed scanline encodings independently exercise all five PNG filters.
scratch=os.path.abspath(sys.argv[1]) if len(sys.argv)>1 else os.path.join(os.getcwd(),"png-helper-checks")
os.makedirs(scratch,exist_ok=True)
def chunk(kind,data):
    return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data)&0xffffffff)
for filtering,row in enumerate([
    [0,0,255,255,255,255], [0,0,255,255,255,0], [1,0,255,255,0,255],
    [129,0,255,255,128,128], [1,0,255,255,0,0]]):
    filtered=encoded[:33]+chunk(b"IDAT",zlib.compress(b"\x00"+pixels[:6]+bytes([filtering])+bytes(row)))+chunk(b"IEND",b"")
    path=os.path.join(scratch,"filter-"+str(filtering)+".png")
    with open(path,"wb") as target: target.write(filtered)
    width,height,decoded=read_png_rgb(path)
    assert (width,height)==(2,2) and bytes(decoded)==pixels,(filtering,decoded)

original=tensor.tensor([1,2,3,4,5,6],shape=[2,3],dtype=tensor.float32)
view=garnet.tensor_views().reshape(original,[1,3,2])
del original
assert list(view.shape)==[1,3,2] and view.tolist()==[1,2,3,4,5,6]
failed=False
try: garnet.tensor_views().reshape(view,[7])
except Exception: failed=True
assert failed
window=garnet.tensor_views().slice(view,2,[2,2])
del view
assert window.tolist()==[3,4,5,6] and list(window.shape)==[2,2]
nested=garnet.tensor_views().slice(window,1,[2])
del window
assert nested.tolist()==[4,5]
for start,shape in [(-1,[1]),(1,[2]),(0,[3])]:
    failed=False
    try: garnet.tensor_views().slice(nested,start,shape)
    except Exception: failed=True
    assert failed,(start,shape)
cosine,sine=rope_tables(2,3,4,{"axes_dims_rope":[2,2,2,2],"attention_head_dim":8,"rope_theta":2000})
assert list(cosine.shape)==[1,10,1,8]
assert cosine.tolist()[:8]==[1]*8 and sine.tolist()[:8]==[0]*8
print("generation schedule, noise, PNG, tensor views and latent-coordinate checks passed",flush=True)
