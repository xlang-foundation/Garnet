"""Append benchmark records once; assemble reports only after model workers close.

The writer retains counters and its file handle, never previous row payloads.
Incomplete journals remain evidence and cannot be collected as completed runs.
This changes evidence IO, not the Model.infer_batch timer or inference behavior.
"""
import json
from pathlib import Path


class JournalError(ValueError):
    pass


def _invalid_constant(value):
    raise JournalError('Non-finite JSON constant '+value)


class StreamEvidence:
    def __init__(self,path,header):
        if not isinstance(header,dict) or any(k in header for k in ('rows','events','period_owners')):
            raise JournalError('Header must exclude accumulated result containers')
        self.sequence=0;self.row_count=0;self.event_count=0;self.closed=False
        self.stream=Path(path).open('x',encoding='utf-8',newline='\n')
        try:self._append('header',header)
        except:
            self.stream.close();self.closed=True
            raise

    def _append(self,kind,payload):
        if self.closed:raise JournalError('Journal is closed')
        # Serialize before writing. An invalid payload cannot add a partial line.
        text=json.dumps({'sequence':self.sequence,'kind':kind,'payload':payload},ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
        written=self.stream.write(text)
        if written!=len(text):raise OSError('Incomplete journal write')
        self.stream.flush()
        self.sequence+=1

    def row(self,payload):
        if not isinstance(payload,dict):raise JournalError('Row must be a dictionary')
        self._append('row',payload);self.row_count+=1

    def event(self,payload):
        if not isinstance(payload,dict):raise JournalError('Event must be a dictionary')
        self._append('event',payload);self.event_count+=1

    def period_owner(self,payload):
        if not isinstance(payload,dict):raise JournalError('Period owner must be a dictionary')
        self._append('period_owner',payload)

    def finish(self,completed,error=None):
        if type(completed) is not bool or error is not None and not isinstance(error,str) or completed and error is not None:
            raise JournalError('Invalid completion status')
        self._append('footer',{'completed':completed,'error':error,'row_count':self.row_count,'event_count':self.event_count})
        self.stream.close();self.closed=True

    def abort(self):
        if not self.closed:self.stream.close();self.closed=True


def collect_journal(path,require_completed=False):
    """Offline collector. Caller must establish worker closure before calling."""
    header=None;footer=None;rows=[];events=[];owners=[];sequence=0
    with Path(path).open('r',encoding='utf-8') as stream:
        for line_number,line in enumerate(stream,1):
            if not line.endswith('\n'):raise JournalError('Truncated journal line '+str(line_number))
            try:record=json.loads(line,parse_constant=_invalid_constant)
            except Exception as error:raise JournalError('Invalid JSON line '+str(line_number)+': '+str(error))
            if not isinstance(record,dict) or set(record)!=set(('sequence','kind','payload')):
                raise JournalError('Invalid journal envelope')
            if type(record['sequence']) is not int or record['sequence']!=sequence:
                raise JournalError('Out-of-order journal sequence')
            sequence+=1
            kind=record['kind'];payload=record['payload']
            if footer is not None:raise JournalError('Records after journal footer')
            if header is None:
                if kind!='header' or not isinstance(payload,dict) or any(k in payload for k in ('rows','events','period_owners')):
                    raise JournalError('Missing or invalid journal header')
                header=payload
            elif kind=='row':
                if not isinstance(payload,dict):raise JournalError('Row must be a dictionary')
                rows.append(payload)
            elif kind=='event':
                if not isinstance(payload,dict):raise JournalError('Event must be a dictionary')
                events.append(payload)
            elif kind=='period_owner':
                if not isinstance(payload,dict):raise JournalError('Period owner must be a dictionary')
                owners.append(payload)
            elif kind=='footer':
                if not isinstance(payload,dict) or set(payload)!=set(('completed','error','row_count','event_count')):
                    raise JournalError('Invalid journal footer')
                if type(payload['completed']) is not bool or payload['error'] is not None and not isinstance(payload['error'],str) or payload['completed'] and payload['error'] is not None:
                    raise JournalError('Invalid journal completion')
                if type(payload['row_count']) is not int or payload['row_count']!=len(rows) or type(payload['event_count']) is not int or payload['event_count']!=len(events):
                    raise JournalError('Journal counts do not match records')
                footer=payload
            else:raise JournalError('Unexpected journal record kind')
    if header is None or footer is None:raise JournalError('Unfinished journal')
    if require_completed and not footer['completed']:raise JournalError('Journal records a failed run')
    result=dict(header);result.update(rows=rows,events=events,completed=footer['completed'],error=footer['error'])
    if owners:result['period_owners']=owners
    return result
