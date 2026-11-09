"""Audit the complete process-tagged Shader Timeline; never starts GPU work.
Usage: python3 analyze_shader_timeline2.py ABSOLUTE_STUDY [TARGET_PID=5973]
Sample Duration is sampled work, not kernel elapsed time or a serving cost share.
"""
from pathlib import Path
import collections
import copy
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET


def parse(payload, target):
    root=ET.fromstring(payload)
    assert root.tag=='trace-query-result' and len(root)==1
    table=root[0];schema=table.find('schema')
    assert schema.attrib['name']=='metal-shader-profiler-intervals'
    columns=[(c.findtext('mnemonic'),c.findtext('name')) for c in schema]
    indices={name:i for i,(name,label) in enumerate(columns)}
    assert columns[indices['duration']][1]=='Sample Duration'
    ids={}
    for e in root.iter():
        key=e.attrib.get('id')
        if key:
            assert key not in ids,'Duplicate public XML ID'
            ids[key]=e
    def resolve(e):
        seen=set()
        while 'ref' in e.attrib:
            key=e.attrib['ref'];assert key not in seen and key in ids,'Bad public XML reference'
            seen.add(key);e=ids[key]
        return e
    result=[]
    for row in table.findall('row'):
        assert len(row)==len(columns)
        get=lambda name:resolve(row[indices[name]])
        process=get('process');assert process.tag=='process'
        pid=process.find('pid');assert pid is not None and int(resolve(pid).text)==target,'Foreign process row'
        start,duration=get('start'),get('duration')
        assert start.tag=='start-time' and duration.tag=='duration'
        start,duration=int(start.text),int(duration.text)
        assert start>=0 and duration>0
        name=get('name');assert name.tag=='metal-object-label' and name.text
        result.append(dict(start_ns=start,sample_duration_ns=duration,shader=name.text))
    assert result
    return result


def audit(study,target):
    load=lambda n:json.loads((study/n).read_text())
    status=load('export-shader-status3.json')
    assert status['status']=='terminal' and status['returncode']==0
    assert load('status1.json')['final_returncode']==0
    assert load('native-output-audit2.json')['passed']
    raw=(study/'shader-export3.xml').read_bytes()
    rows=parse(raw,target)
    retained=(study/'shader-timeline1.xml').read_bytes()
    digest=hashlib.sha256(retained).hexdigest()
    assert digest==load('export-query-scope-reduction1.json')['complete_shader_section_sha256']
    assert rows==parse(retained,target),'Narrow reexport differs from retained complete table'
    old=load('shader-timeline-rows1.json')
    assert rows==[{k:r[k] for k in ('start_ns','sample_duration_ns','shader')} for r in old]
    assert len(rows)==2906
    # Negative controls exercise actual public-reference/process/schema validation.
    root=ET.fromstring(raw);rejected=0
    for kind in ('reference','process','duration','schema'):
        changed=copy.deepcopy(root)
        if kind=='reference':
            e=changed.find('node/row')[2];e.attrib.clear();e.attrib['ref']='missing'
        elif kind=='process':
            e=next(e for e in changed.iter('pid') if e.attrib.get('id'));e.text=str(target+1)
        elif kind=='duration':
            e=next(e for e in changed.iter('duration') if e.attrib.get('id'));e.text='-1'
        else:
            e=changed.find('node/schema/col[name="Sample Duration"]/name');e.text='Elapsed kernel duration'
        try:parse(ET.tostring(changed),target)
        except (AssertionError,ValueError):rejected+=1
        else:raise AssertionError('Invalid XML accepted: '+kind)
    families=collections.Counter();counts=collections.Counter()
    for row in rows:
        family=re.sub(r' \(\d+\)$','',row['shader'])
        families[family]+=row['sample_duration_ns'];counts[family]+=1
    total=sum(families.values());assert total==942702088
    return dict(passed=True,scope=__doc__,target_pid=target,rows=len(rows),
        complete_reexport_sha256=hashlib.sha256(raw).hexdigest(),retained_complete_table_sha256=digest,
        narrow_reexport_equals_retained_rows=True,negative_controls_rejected=rejected,
        summed_sample_duration_ns=total,
        shader_families=[dict(shader=n,rows=counts[n],sample_duration_ns=t,share_of_summed_samples_percent=100*t/total) for n,t in families.most_common()],
        original_broad_export_intentionally_cancelled=load('export-gpu-status2.json')['returncode']==1,
        original_broad_partial_present=(study/'gpu-tables2.xml').exists(),
        full_device_counter_aggregate_available=False,
        attribution='Explicit process ownership for shader samples. No attribution of global hardware counters, kernel elapsed cost fractions, theoretical peak or serving speed.',
        serving_speed_claim=False,quality_gate_passed=False,goal_complete=False)


if __name__=='__main__':
    study=Path(sys.argv[1]).resolve();target=int(sys.argv[2]) if len(sys.argv)>2 else 5973
    result=audit(study,target)
    (study/'shader-timeline-audit2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
