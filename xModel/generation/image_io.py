# SPDX-License-Identifier: Apache-2.0
"""PNG I/O through XLang3's native struct/zlib modules; no imaging framework."""
import os
import struct
import zlib

def png_rgb(width,height,pixels):
    if width<1 or height<1 or len(pixels)!=width*height*3:
        raise ValueError("PNG RGB dimensions and byte count differ")
    def chunk(kind,data):
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data)&0xffffffff)
    stride=width*3
    raw=b"".join(b"\x00"+bytes(pixels[row*stride:(row+1)*stride]) for row in range(height))
    return b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b"")

def save_rgb_tensor(image,path):
    import garnet
    target=os.path.abspath(path) if path else ''
    return garnet.media().encode_image(image,target,'{"format":"png","provider":"image"}')

def read_png_rgb(path):
    """Read noninterlaced 8-bit RGB/RGBA PNG, compositing alpha over white."""
    with open(path,"rb") as source: encoded=source.read()
    if encoded[:8]!=b"\x89PNG\r\n\x1a\n": raise ValueError("reference image must be a PNG file")
    offset=8
    width=height=channels=0
    compressed=[]
    ended=False
    while offset<len(encoded):
        if offset+12>len(encoded): raise ValueError("truncated PNG chunk")
        size=struct.unpack(">I",encoded[offset:offset+4])[0]
        kind=encoded[offset+4:offset+8]
        end=offset+8+size
        if end+4>len(encoded): raise ValueError("truncated PNG data")
        data=encoded[offset+8:end]
        crc=struct.unpack(">I",encoded[end:end+4])[0]
        if zlib.crc32(kind+data)&0xffffffff!=crc: raise ValueError("PNG checksum mismatch")
        if kind==b"IHDR":
            if offset!=8 or size!=13: raise ValueError("invalid PNG header")
            width,height,depth,color,compression,filtering,interlace=struct.unpack(">IIBBBBB",data)
            if not (0<width<=4096 and 0<height<=4096) or depth!=8 or color not in (2,6) or compression or filtering or interlace:
                raise ValueError("PNG requires noninterlaced RGB/RGBA 8-bit pixels up to 4096x4096")
            channels=3 if color==2 else 4
        elif kind==b"IDAT":
            if not channels: raise ValueError("PNG data precedes header")
            compressed.append(data)
        elif kind==b"IEND":
            if size: raise ValueError("invalid PNG end")
            ended=True
            break
        elif kind[0]&32==0: raise ValueError("unsupported critical PNG chunk")
        offset=end+4
    if not ended or not channels or not compressed: raise ValueError("incomplete PNG")
    raw=zlib.decompress(b"".join(compressed))
    stride=width*channels
    if len(raw)!=(stride+1)*height: raise ValueError("PNG pixel data size mismatch")
    pixels=bytearray()
    previous=bytearray(stride)
    for y in range(height):
        start=y*(stride+1)
        filtering=raw[start]
        if filtering>4: raise ValueError("invalid PNG scanline filter")
        row=bytearray(raw[start+1:start+1+stride])
        # Filter zero already contains reconstructed bytes. Keep previous=row
        # below so a following Up/Average/Paeth row sees the same history.
        if filtering:
            for x in range(stride):
                left=row[x-channels] if x>=channels else 0
                above=previous[x]
                corner=previous[x-channels] if x>=channels else 0
                predictor=0
                if filtering==1: predictor=left
                elif filtering==2: predictor=above
                elif filtering==3: predictor=(left+above)//2
                elif filtering==4:
                    p=left+above-corner
                    a,b,c=abs(p-left),abs(p-above),abs(p-corner)
                    predictor=left if a<=b and a<=c else (above if b<=c else corner)
                row[x]=(row[x]+predictor)&255
        if channels==3: pixels.extend(row)
        else:
            for x in range(width):
                alpha=row[4*x+3]
                pixels.extend((row[4*x+c]*alpha+255*(255-alpha)+127)//255 for c in range(3))
        previous=row
    return width,height,pixels

def load_rgb_tensor(path):
    import garnet
    return garnet.media().decode_image(path if isinstance(path,bytes) else os.path.abspath(path))
