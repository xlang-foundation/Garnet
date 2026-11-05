#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned MLX functional reference; no serving or comparative speed claim."""
import argparse,hashlib,importlib.metadata,json,pathlib
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=pathlib.Path,required=True)
 p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--max-tokens',type=int,default=32)
 a=p.parse_args();a.output.mkdir(exist_ok=False)
 model,tokenizer=load(str(a.model),tokenizer_config={'trust_remote_code':False})
 text='Write a Python function add(a, b) that returns a + b. Output only code.'
 rendered=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True)
 ids=tokenizer.encode(rendered,add_special_tokens=False);cache=make_prompt_cache(model);generated=[]
 tokens=mx.array([ids]);stops=set([tokenizer.eos_token_id]);logits_files=[]
 for step in range(a.max_tokens):
  logits=model(tokens,cache=cache)[:,-1,:];next_token=mx.argmax(logits,axis=-1);mx.eval(logits,next_token)
  value=int(next_token.item());generated.append(value)
  if step==0:
   mx.save(str(a.output/'first-logits.npy'),logits.astype(mx.float32));logits_files.append('first-logits.npy')
  if value in stops:break
  tokens=mx.array([[value]])
 result=dict(scope='Small functional greedy reference; not a vLLM/SGLang baseline',model=str(a.model),prompt=text,rendered=rendered,prompt_ids=ids,max_tokens=a.max_tokens,generated_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=True),stops=list(stops),versions={x:importlib.metadata.version(x) for x in ['mlx','mlx-lm','transformers']},logits_files=logits_files)
 (a.output/'reference.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
