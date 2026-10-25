"""Explicit benchmark stop policy; default natural stopping remains unchanged."""
def generation_stop_ids(body,stop_ids):
 value=body.get('ignore_eos',False)
 if type(value) is not bool:raise ValueError('ignore_eos requires an explicit boolean')
 return [] if value else list(stop_ids)
