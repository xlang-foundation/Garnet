# SPDX-License-Identifier: Apache-2.0
"""Validate native prompt formatting, truncation, padding and causal masking."""
import json
import os
import sys
sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","..")))
from xModel.flux2.klein_4b.prompt import prompt_inputs

with open(sys.argv[1],"r",encoding="utf-8") as source: reference=json.load(source)
for index,case in enumerate(reference["cases"]):
    inputs,count=prompt_inputs(sys.argv[2],case["prompt"],512)
    assert inputs[0].tolist()==case["ids"],("prompt token mismatch",index)
    assert count==sum(case["attention_mask"]),(index,count)
    mask=inputs[3].tolist()
    for query in range(512):
        for key in range(512):
            expected=key<=query and case["attention_mask"][key]==1
            assert (mask[query*512+key]==0)==expected,(index,query,key)
print("FLUX checkpoint template, padding, truncation and causal masking parity passed",len(reference["cases"]),flush=True)
