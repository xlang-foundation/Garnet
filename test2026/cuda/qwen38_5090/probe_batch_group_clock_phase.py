"""CPU four-row deadline experiment, including a clock tick between arrivals."""
from pathlib import Path
import hashlib,json,platform,queue,statistics,sys,threading,time

output=Path(sys.argv[1]);assert not output.exists();rows=[]
for mode in ('legacy_monotonic_break','precise_perf_counter_retry'):
 clock=time.monotonic if mode.startswith('legacy') else time.perf_counter
 for interval in (.001,.003):
  for trial in range(256):
   jobs=queue.Queue();release=threading.Event();puts=[]
   def feed():
    release.wait()
    for row in (1,2,3):
     time.sleep(interval);before=time.perf_counter();jobs.put(row);puts.append({'row':row,'started':before,'finished':time.perf_counter()})
   feeder=threading.Thread(target=feed);feeder.start()
   started=time.perf_counter();deadline=clock()+.010;release.set();group=[0];empty=0
   while len(group)<4:
    remaining=deadline-clock()
    if remaining<=0:break
    try:ticket=jobs.get(timeout=remaining)
    except queue.Empty:
     empty+=1
     if mode.startswith('legacy'):break
     continue
    group.append(ticket)
   dispatched=time.perf_counter();feeder.join(timeout=5);assert not feeder.is_alive() and len(puts)==3
   rows.append({'mode':mode,'interval_seconds':interval,'trial':trial,'started':started,'dispatched':dispatched,'elapsed_seconds':dispatched-started,'included_rows':group,'puts':puts,'empty_returns':empty,
                'missed_rows_put_finished_before_true_10ms':[p['row'] for p in puts if p['finished']-started<.010 and p['row'] not in group]})
summaries=[]
for mode in ('legacy_monotonic_break','precise_perf_counter_retry'):
 for interval in (.001,.003):
  selected=[r for r in rows if r['mode']==mode and r['interval_seconds']==interval]
  summaries.append({'mode':mode,'interval_seconds':interval,'trials':len(selected),'full_groups':sum(len(r['included_rows'])==4 for r in selected),'early_partial_before_9ms':sum(len(r['included_rows'])<4 and r['elapsed_seconds']<.009 for r in selected),
                    'trials_missing_put_before_true_10ms':sum(bool(r['missed_rows_put_finished_before_true_10ms']) for r in selected),'minimum_elapsed_seconds':min(r['elapsed_seconds'] for r in selected),'median_elapsed_seconds':statistics.median(r['elapsed_seconds'] for r in selected)})
result={'status':'cpu_four_row_group_clock_phase_observed','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'platform':platform.platform(),
        'clocks':{name:vars(time.get_clock_info(name)) for name in ('monotonic','perf_counter')},'wait_seconds':.010,'capacity':4,'summaries':summaries,'raw_trials':rows,
        'scope':'CPU queue/deadline experiment only; original versus QPC/retry loop, four arrivals. All1024trials retained. No HTTP/model/GPU measurement and no claim this alone explains an earlier HTTP regression.'}
output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='raw_trials'},indent=2))
