"""Offline report assembly; controller calls this only after its worker exits."""
import argparse,json
from pathlib import Path
from qwen35_stream_evidence import collect_journal
p=argparse.ArgumentParser()
p.add_argument('journal',type=Path);p.add_argument('output',type=Path)
p.add_argument('--expected-rows',type=int,required=True)
a=p.parse_args();assert not a.output.exists(),'Preserve original collected report'
j=collect_journal(a.journal)
assert len(j['rows'])==a.expected_rows or not j['completed']
assert j['events'] and j['events'][-1]['name']=='closed'
with a.output.open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
print('CLOSED_WORKER_STREAM_COLLECTED',len(j['rows']),j['completed'])
