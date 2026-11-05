"""Run with the native XLang3 runtime, without Python tokenization libraries."""
# SPDX-License-Identifier: Apache-2.0
import json,sys
sys.path.insert(0,sys.argv[1])
import garnet
with open(sys.argv[2]) as stream:fixtures=json.load(stream)
tokenizer=garnet.tokenizer()
checks=0
for fixture in fixtures:
 assert tokenizer.open(fixture['folder'])
 info=tokenizer.info()
 assert info['family']==fixture['family'];checks+=1
 for case in fixture['cases']:
  assert tokenizer.encode(case['text'])==case['plain'];checks+=1
  assert tokenizer.encode_ex(case['text'],True)==case['special'];checks+=1
  assert tokenizer.decode(case['plain'])==case['decoded'];checks+=1
  assert tokenizer.decode_ex(case['special'],True)==case['skip_decoded'];checks+=1
 assert tokenizer.token_id(fixture['special_token'])==fixture['special_id'];checks+=1
 failed=False
 try:tokenizer.open(fixture['folder']+'/missing-tokenizer-directory')
 except Exception as error:failed=True
 assert failed;checks+=1
 failed=False
 try:tokenizer.encode('must not reuse the previous tokenizer')
 except Exception as error:failed=True
 assert failed;checks+=1
print('native_tokenizer_runtime_api',json.dumps(dict(passed=True,models=len(fixtures),checks=checks)))
