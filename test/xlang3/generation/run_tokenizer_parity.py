# SPDX-License-Identifier: Apache-2.0
"""Compare Garnet's native C++ tokenizer with independent checkpoint fixtures."""
import json
import os
import sys
sys.path.insert(0,os.getcwd())
import garnet

with open(sys.argv[1],"r",encoding="utf-8") as source: reference=json.load(source)
tokenizer=garnet.tokenizer()
tokenizer.open(sys.argv[2])
failures=[]
for index,case in enumerate(reference["cases"]):
    actual=list(tokenizer.encode_ex(case["text"],case.get("add_special_tokens",False)))
    if actual!=case["ids"]:
        failures.append({"index":index,"text":case["text"],"expected":case["ids"],"actual":actual,"stage":"encode"})
        continue
    for skip,key in [(False,"decoded"),(True,"decoded_skip")]:
        if key in case:
            decoded=tokenizer.decode_ex(actual,skip)
            if decoded!=case[key]: failures.append({"index":index,"text":case["text"],"stage":key,"expected":case[key],"actual":decoded})
summary={"reference":reference["reference"],"reference_version":reference["version"],
    "tokenizer":tokenizer.info(),"cases":len(reference["cases"]),"failures":len(failures),"examples":failures[:5]}
print(json.dumps(summary,ensure_ascii=False),flush=True)
if len(sys.argv)>3:
    with open(sys.argv[3],"w",encoding="utf-8") as output: json.dump(summary,output,ensure_ascii=False)
assert not failures,"native tokenizer differs from independent checkpoint fixtures"
