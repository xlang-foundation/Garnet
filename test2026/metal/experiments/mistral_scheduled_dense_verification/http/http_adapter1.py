"""Pass original user text to bounded drafting; native runtime owns decoding."""
from local_coder_batch_http import MetalBatchWorker,BatchHandler
from local_coder_batch_transport import BatchWorker,RequestRejected
class ScheduledWorker(MetalBatchWorker):
 def generate(self,body,key):
  user=body['messages'][0]['content']
  if not isinstance(user,str):raise RequestRejected('Plain user text required')
  try:rendered=self.template.render(messages=body['messages'],add_generation_prompt=True,enable_thinking=False,tools=None,**self.special_tokens)
  except Exception as error:raise RequestRejected(str(error)) from error
  yield from BatchWorker.generate(self,dict(rendered_prompt=rendered,draft_user_content=user,max_tokens=body['max_tokens']),key)
class ScheduledHandler(BatchHandler):
 def validate_request(self,body):
  super().validate_request(body)
  if body['max_tokens']>128:raise ValueError('Diagnostic B1 profile output limit128')
