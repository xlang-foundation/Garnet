# SPDX-License-Identifier: Apache-2.0
"""Run with XLang3: generate_flux2.py CHECKPOINT CACHE OUTPUT PROMPT [SIZE] [SEED] [REFERENCE_PNG].

Configure the hardware backend in this worker's environment before launching.
For bounded TensorRT residency, set GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB.
"""
import json
import os
import sys
sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))
import garnet
from xModel.generation.cli import image_size

if len(sys.argv) not in (5,6,7,8): raise ValueError(__doc__)
height,width=image_size(sys.argv[5] if len(sys.argv)>5 else "512")
seed=int(sys.argv[6]) if len(sys.argv)>6 else 0
def report(message): print(message,flush=True)
result=garnet.generate("flux2-klein-4b",sys.argv[4],checkpoint=sys.argv[1],cache=sys.argv[2],output=sys.argv[3],height=height,width=width,seed=seed,progress=report,
    reference_image=sys.argv[7] if len(sys.argv)>7 else None)
with open(os.path.abspath(sys.argv[3])+".json","w") as target: json.dump(result,target)
print(json.dumps(result),flush=True)
