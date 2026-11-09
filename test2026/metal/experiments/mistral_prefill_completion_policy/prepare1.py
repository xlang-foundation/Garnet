"""Prepare an opt-in native completion-first policy from the frozen engine."""
from pathlib import Path
import difflib,hashlib,json,subprocess,shutil,sys
PACKAGE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve()
assert ROOT.parent.name=='mistral7b-local-coder' and not ROOT.exists()
ROOT.mkdir()
for name in ['build_native1.py', 'prepare_admission1.py', 'run_admission1.py', 'audit_saved_scheduling1.py', 'audit_admission1.py', 'prepare1.py']:shutil.copyfile(PACKAGE/name,ROOT/name)
M=ROOT.parent;G=M.parents[2]/'Garnet';N=M/'mistral-fused-mlp-native1';S=ROOT/'source1'
subprocess.run([sys.executable,'-B',str(ROOT/'audit_saved_scheduling1.py')],check=True)

read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    assert not p.exists(),str(p);p.write_text(json.dumps(v,indent=2)+'\n')
assert read(ROOT/'saved-scheduling-audit1.json')['passed'] and not S.exists()
assert sha(N/'provenance1.json')=='b94f6e7ee34d17079439e88cd0d5e419ff9c11e8d2514fbe2dbd00f589917ebe'
original=read(N/'source-bindings2.json')['files_sha256']
for name,h in original.items():assert sha(N/'source2'/name)==h,name
dirty=read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256']
for name,h in dirty.items():assert sha(G/name)==h,name
subprocess.run(['cp','-cR',str(N/'source2'),str(S)],check=True)
changes={}
def edit(name,old,new):
    p=S/name;text=p.read_text();assert text.count(old)==1,(name,old[:70],text.count(old))
    p.chmod(0o644);p.write_text(text.replace(old,new));changes.setdefault(name,[]).append(dict(old=old,new=new))
header='src/runtime/scheduler/continuous_batch_scheduler.h';core='src/runtime/scheduler/continuous_batch_scheduler.cpp'
edit(header,'        // Optional backend profile choice after row placement/prefix restore.',
'''        // Opt-in FIFO completion policy: one bounded prefill owner at a time.
        // Ready decode batches still run first. The default rotates all owners.
        bool completePrefillFirst = false;
        // Optional backend profile choice after row placement/prefix restore.''')
edit(header,'        bool SetPrefillChunkTokens(int tokens);','''        bool SetPrefillChunkTokens(int tokens);
        bool SetPrefillCompletionPriority(bool enabled);
        bool PrefillCompletionPriority() const {return m_config.completePrefillFirst;}''')
edit(core,'        std::sort(m_config.decodeBuckets.begin(), m_config.decodeBuckets.end());','''        if(m_config.completePrefillFirst && !m_config.prefillChunkTokens)
            throw std::invalid_argument("Completion-first prefill requires bounded chunks");
        std::sort(m_config.decodeBuckets.begin(), m_config.decodeBuckets.end());''')
edit(core,'                   m_config.maxPrefillBatch) {','                   (m_config.completePrefillFirst ? std::min(1,m_config.maxPrefillBatch) : m_config.maxPrefillBatch)) {')
edit(core,'        if(!m_requests.empty()||chunkTokens<1||chunks<2||chunks>16||tokenBudget<1||','        if(m_config.completePrefillFirst||!m_requests.empty()||chunkTokens<1||chunks<2||chunks>16||tokenBudget<1||')
edit(core,'        state.prefillScheduled=0;state.cachedHeadScheduled=false;m_prefillQueue.push_back(id);return true;','''        state.prefillScheduled=0;state.cachedHeadScheduled=false;
        if(m_config.completePrefillFirst)m_prefillQueue.push_front(id);else m_prefillQueue.push_back(id);
        return true;''')
edit(core,'        if (tokens < 0 || tokens > m_config.maxPrefillTokensPerTick || !m_requests.empty()) return false;','        if (tokens < 0 || tokens > m_config.maxPrefillTokensPerTick || !m_requests.empty() || (m_config.completePrefillFirst && !tokens)) return false;')
edit(core,'    bool ContinuousBatchScheduler::CompletePrefillChunk(','    bool ContinuousBatchScheduler::SetPrefillCompletionPriority(bool enabled)\n    {\n        if(!m_requests.empty() || (enabled && (!m_config.prefillChunkTokens || m_groupChunks>1)))return false;\n        m_config.completePrefillFirst=enabled;return true;\n    }\n\n    bool ContinuousBatchScheduler::CompletePrefillChunk(')
edit(core,'        m_prefillQueue.push_back(requestId);\n        return true;','        if(m_config.completePrefillFirst)m_prefillQueue.push_front(requestId);else m_prefillQueue.push_back(requestId);\n        return true;')
bridge='src/backends/metal/continuous_scheduler.h';bridgecpp='src/backends/metal/continuous_scheduler.cpp'
edit(bridge,'    bool SetPrefillChunk(int64_t tokens);','    bool SetPrefillChunk(int64_t tokens);\n    bool SetPrefillCompletionPriority(bool enabled);')
edit(bridge,'        APISET().AddFunc<1>("set_prefill_chunk",&MetalContinuousScheduler::SetPrefillChunk);','        APISET().AddFunc<1>("set_prefill_chunk",&MetalContinuousScheduler::SetPrefillChunk);\n        APISET().AddFunc<1>("set_prefill_completion_priority",&MetalContinuousScheduler::SetPrefillCompletionPriority);')
edit(bridgecpp,'bool MetalContinuousScheduler::SetPrefillChunk(int64_t tokens) {','''bool MetalContinuousScheduler::SetPrefillCompletionPriority(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    if(!inflight_.empty())return false;
    return scheduler_->SetPrefillCompletionPriority(enabled);
}
bool MetalContinuousScheduler::SetPrefillChunk(int64_t tokens) {''')
edit(bridgecpp,'{"tracked",scheduler_->TrackedRequestCount()},{"inflight",inflight_.size()},{"rows",rows_},{"capacity",capacity_}',
    '{"tracked",scheduler_->TrackedRequestCount()},{"inflight",inflight_.size()},{"rows",rows_},{"capacity",capacity_},{"prefill_completion_priority",scheduler_->PrefillCompletionPriority()}')
edit('src/entry/metal_main.cpp','; native_request_row_leases; native_continuous_scheduler; chunked_prefill_scheduler; packed_token_plan','; native_request_row_leases; native_continuous_scheduler; chunked_prefill_scheduler; packed_token_plan; native_prefill_completion_priority_optional')
test=r'''
void CompletionPriorityChecks() {
    ContinuousBatchSchedulerConfig config;config.decodeBuckets={1,2,4};config.maxPrefillBatch=4;
    config.maxPrefillTokensPerTick=16;config.prefillChunkTokens=4;
    GenerationRequest a;a.promptTokens=11;a.maxNewTokens=3;
    GenerationRequest b=a;b.promptTokens=7;GenerationRequest c=a;c.promptTokens=3;
    // Disabled policy retains the existing three-owner round-robin plan.
    FixedRows oldRows;ContinuousBatchScheduler old(config);
    old.Submit(a);old.Submit(b);old.Submit(c);
    auto oldTick=old.Schedule(oldRows);Check(oldTick.batches.size()==1&&oldTick.batches[0].requests.size()==3);
    FixedRows rows;ContinuousBatchScheduler policy(config);
    Check(!policy.PrefillCompletionPriority()&&policy.SetPrefillCompletionPriority(true));
    Check(policy.PrefillCompletionPriority()&&!policy.SetPrefillChunkTokens(0));
    Check(!policy.SetPrefillGrouping(4,2,32));
    auto ai=policy.Submit(a),bi=policy.Submit(b),ci=policy.Submit(c);
    Check(ai&&bi&&ci&&!policy.SetPrefillCompletionPriority(false));
    auto only=[&](const SchedulerTick& tick,RequestStage stage,uint64_t id,int offset,int count) {
        const BatchPlan* selected=nullptr;
        for(const auto& batch:tick.batches)if(batch.stage==stage){Check(selected==nullptr);selected=&batch;}
        Check(selected&&selected->requests.size()==1);
        const auto& r=selected->requests[0];Check(r.id==id&&r.prefillOffset==offset&&r.prefillTokens==count);
        Check(r.requestRow==rows.Placement(id).requestRow&&r.rowEpoch==rows.Placement(id).rowEpoch);
    };
    auto tick=policy.Schedule(rows);only(tick,RequestStage::Prefill,ai,0,4);
    auto ap=rows.Placement(ai);Check(rows.LogicalLength(bi)==-1&&rows.LogicalLength(ci)==-1);
    Check(!policy.CompletePrefillChunk(ai,3,rows)&&policy.CompletePrefillChunk(ai,4,rows));
    tick=policy.Schedule(rows);only(tick,RequestStage::Prefill,ai,4,4);
    Check(rows.Placement(ai).rowEpoch==ap.rowEpoch&&policy.CompletePrefillChunk(ai,4,rows));
    tick=policy.Schedule(rows);only(tick,RequestStage::Prefill,ai,8,3);
    Check(!policy.CompletePrefillChunk(ai,3,rows)&&policy.CompletePrefill(ai,41,rows));
    tick=policy.Schedule(rows);Check(tick.batches.size()==2&&tick.batches[0].stage==RequestStage::Decode);
    Check(tick.batches[0].requests[0].id==ai);only(tick,RequestStage::Prefill,bi,0,4);
    auto bp=rows.Placement(bi);Check(bp.requestRow!=ap.requestRow);
    Check(policy.CompleteDecode(ai,42,false,rows)&&policy.CompletePrefillChunk(bi,4,rows));
    tick=policy.Schedule(rows);only(tick,RequestStage::Prefill,bi,4,3);
    Check(rows.Placement(bi).rowEpoch==bp.rowEpoch);
    Check(policy.CompleteDecode(ai,43,false,rows)&&rows.LogicalLength(ai)==-1);
    Check(policy.CompletePrefill(bi,51,rows));
    tick=policy.Schedule(rows);Check(tick.batches[0].stage==RequestStage::Decode);
    only(tick,RequestStage::Prefill,ci,0,3);auto cp=rows.Placement(ci);
    Check(cp.requestRow==ap.requestRow&&cp.rowEpoch!=ap.rowEpoch);
    Check(policy.CompleteDecode(bi,52,false,rows)&&policy.CompletePrefill(ci,61,rows));
    tick=policy.Schedule(rows);Check(tick.batches.size()==1&&tick.batches[0].requests.size()==2);
    Check(policy.Cancel(bi,rows)&&policy.Cancel(ci,rows));
    Check(policy.ForgetFinished(ai)&&policy.ForgetFinished(bi)&&policy.ForgetFinished(ci));
    Check(policy.SetPrefillCompletionPriority(false)&&!policy.PrefillCompletionPriority());
    // Abandon/retry keeps the current owner; cancellation admits the next owner.
    ai=policy.Submit(a);bi=policy.Submit(b);Check(policy.Cancel(ai,rows)&&policy.Cancel(bi,rows));
    Check(policy.ForgetFinished(ai)&&policy.ForgetFinished(bi)&&policy.SetPrefillCompletionPriority(true));
    ai=policy.Submit(a);bi=policy.Submit(b);
    tick=policy.Schedule(rows);only(tick,RequestStage::Prefill,ai,0,4);
    Check(policy.AbandonPrefillChunk(ai));tick=policy.Schedule(rows);only(tick,RequestStage::Prefill,ai,0,4);
    Check(policy.Cancel(ai,rows)&&!policy.CompletePrefillChunk(ai,4,rows));
    tick=policy.Schedule(rows);only(tick,RequestStage::Prefill,bi,0,4);Check(policy.Cancel(bi,rows));
    // A full prefix hit still schedules its zero-token prediction step.
    FixedRows cached;ContinuousBatchScheduler hit(config);Check(hit.SetPrefillCompletionPriority(true));
    auto hi=hit.Submit(c);cached.SeedPrefix(hi,3,true);tick=hit.Schedule(cached);
    Check(tick.batches.size()==1&&tick.batches[0].requests[0].prefillTokens==0&&hit.CompletePrefill(hi,71,cached));
    Check(hit.Cancel(hi,cached));
    ContinuousBatchScheduler grouped(config);Check(grouped.SetPrefillGrouping(4,2,32)&&!grouped.SetPrefillCompletionPriority(true));
    ContinuousBatchSchedulerConfig unbounded;ContinuousBatchScheduler whole(unbounded);Check(!whole.SetPrefillCompletionPriority(true));
    unbounded.completePrefillFirst=true;bool rejected=false;
    try{ContinuousBatchScheduler invalid(unbounded);}catch(const std::invalid_argument&){rejected=true;}Check(rejected);
}
'''
edit('test2026/native/scheduler_fixed_rows_test.cpp','int main() {',test+'\nint main() {\n    CompletionPriorityChecks();')
patch=''.join(''.join(difflib.unified_diff((N/'source2'/name).read_text().splitlines(True),(S/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name)) for name in sorted(changes))
(ROOT/'candidate1.patch').write_text(patch)
current={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
assert set(current)==set(original)
changed=sorted(name for name,h in current.items() if h!=original[name]);assert changed==sorted(changes),changed
save(ROOT/'source-bindings1.json',dict(passed=True,source=str(S),base_source_bindings_sha256=sha(N/'source-bindings2.json'),files_sha256=current,changed_paths=changed,changes=changes,patch_sha256=sha(ROOT/'candidate1.patch')))
command=read(N/'configure2-command.json');command=[str(S) if v==str(N/'source2') else str(ROOT/'build1') if v==str(N/'build2') else v for v in command]
save(ROOT/'configure1-command.json',command)
save(ROOT/'plan1.json',dict(scope=__doc__,previous_goal_turn='progress: frozen scheduling audit established clustered native first tokens and staggered frozen reference first tokens; comparator kernels/schedules are not dynamically observed.',
    policy_default=False,per_owner_chunk=128,global_prefill_budget=512,decode_priority_unchanged=True,
    eligibility='Explicit bounded chunks; configure only with no tracked requests; incompatible with grouped bulk prefill. Resume current prefill owner first, retaining leased rows/epochs and prefix restore.',
    benchmark_gate='First prove portable lifecycle and unchanged default tests, then native model/cache lineage and exact output IDs/text before any uninstrumented serving comparison. Report median/tail TTFT, decode/inter-token latency and aggregate throughput together.',
    original_dirty_files_preserved=len(dirty),native_base_provenance_sha256=sha(N/'provenance1.json'),source_bindings_sha256=sha(ROOT/'source-bindings1.json'),no_comparator_execution=True,goal_complete=False))
print(json.dumps(dict(prepared=True,source_files=len(current),changed_paths=changed,patch_sha256=sha(ROOT/'candidate1.patch'))),flush=True)

assert sha(ROOT/"candidate1.patch")=="70a4839e9fb16fb7c842c633bfa57ab78b9062c52114efe4ac1aab6a03376129", "Prepared native patch differs"
