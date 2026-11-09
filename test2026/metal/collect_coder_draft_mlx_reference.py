#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone packed coder draft reference; no server or performance campaign."""
import json,sys,time,importlib.metadata
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
inputs=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);assert not output.exists()
model,tokenizer=load(inputs['model']);rows=[]
for case in inputs['cases']:
 ids=tokenizer.encode(case['rendered'],add_special_tokens=False);generated=[];top=None
 for token,probabilities in generate_step(mx.array(ids),model,max_tokens=case['max_tokens']):
  token=int(token);generated.append(token)
  if top is None:
   mx.eval(probabilities);scores=probabilities.tolist();order=sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:8];top=[[i,scores[i]] for i in order]
  if token in inputs['stops']:break
 rows.append(dict(case_id=case['id'],prompt_ids=ids,generated_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=True),first_top8_logprob=top))
 print('mlx_draft_case',case['id'],len(generated),flush=True)
output.write_text(json.dumps(dict(completed=True,inputs=inputs,records=rows,packages={n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers']},scope=__doc__),indent=2)+'\n')
