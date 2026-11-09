"""Close wide preparation for the initialized engine lifetime, not request lifetime."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-wide-dense-verification1/source2';S=R/'source1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((M/'mistral-wide-verification-checkpoint1.json').read_text())['passed'];assert not S.exists()
subprocess.run(['/bin/cp','-cR',str(B),str(S)],check=True)
for p in [S]+list(S.rglob('*')):
 if not p.is_symlink():p.chmod(p.stat().st_mode|0o200)
changed=set()
def edit(name,a,b):
 p=S/name;s=p.read_text();assert s.count(a)==1,(name,s.count(a),a[:70]);p.write_text(s.replace(a,b));changed.add(name)
base='src/backends/metal/';flag='GARNET_METAL_WIDE_DENSE_VERIFICATION'
edit(base+'decoder_session.h','    bool poisoned_=false;','''    bool poisoned_=false;
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    // Reset/release must not reopen compilation of this initialized engine.
    bool wideVerificationPreparationClosed_=false;
#endif''')
edit(base+'decoder_session.h','    bool PrepareVerification(X::Value graphs,X::Value head);','''    bool PrepareVerification(X::Value graphs,X::Value head);
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    bool WideVerificationPreparationOpen(){std::lock_guard<std::mutex> lock(mutex_);return gpu_&&!poisoned_&&!wideVerificationPreparationClosed_;}
#endif''')
edit(base+'decoder_session.h','        APISET().AddFunc<2>("prepare_verification",&MetalDecoderSession::PrepareVerification);','''        APISET().AddFunc<2>("prepare_verification",&MetalDecoderSession::PrepareVerification);
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
        APISET().AddProp("wide_verification_preparation_open",&MetalDecoderSession::WideVerificationPreparationOpen);
#endif''')
edit(base+'decoder_session.cpp','void MetalDecoderSession::Clear() {','''void MetalDecoderSession::Clear() {
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    wideVerificationPreparationClosed_=false;
#endif''')
edit(base+'decoder_session.cpp','        ResetRowUnlocked(int64_t(lease.row));managedRows_=true;return result;','''        ResetRowUnlocked(int64_t(lease.row));managedRows_=true;
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
        wideVerificationPreparationClosed_=true;
#endif
        return result;''')
edit(base+'decoder_session.cpp','    kvInplaceUpdates_=0;constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;poisoned_=true;gpu_->Begin();','''#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    wideVerificationPreparationClosed_=true;
#endif
    kvInplaceUpdates_=0;constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;poisoned_=true;gpu_->Begin();''')
edit(base+'verification_session.cpp','    const auto b=staged->batch,t=staged->chunk,w=staged->width,r=b*t;','''    const auto b=staged->batch,t=staged->chunk,w=staged->width,r=b*t;
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    if(wideVerificationPreparationClosed_&&(r>4||(verification_&&verification_->batch*verification_->chunk>4)))
        throw std::invalid_argument("Prepare or replace a wide verification profile before the first row claim or inference; reset does not reopen preparation");
#endif''')
edit(base+'compact_prefill_session.cpp','    if(!gpu_||poisoned_||managedRows_||compactSealed_||compactRunning_||states_.size()!=32)return false;','''#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    if(wideVerificationPreparationClosed_)return false;
#endif
    if(!gpu_||poisoned_||managedRows_||compactSealed_||compactRunning_||states_.size()!=32)return false;''')
edit(base+'compact_prefill_session.cpp','    if(!gpu_||poisoned_||states_.empty()||states_[0].rows!=4||managedRows_||compactSealed_||','''    if(!gpu_||poisoned_||states_.empty()||states_[0].rows!=4||managedRows_||compactSealed_||
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
       wideVerificationPreparationClosed_||
#endif''')
edit('xModel/mistral/text_7b/metal_decoder.py','        import tensor\n        from xModel.qwen.metal_dense.layers import T,DensePrefillLayer,norm','''        if wide and 'native_wide_verify_preparation_lifetime_v1' in garnet.backend_info() and not self.session.wide_verification_preparation_open:
            raise ValueError('Prepare wide verification before the first claim or inference; reset does not reopen engine preparation')
        import tensor
        from xModel.qwen.metal_dense.layers import T,DensePrefillLayer,norm''')
edit('src/entry/metal_main.cpp','    info+="; native_dense_verify_b4t4_optional_v1";','    info+="; native_dense_verify_b4t4_optional_v1; native_wide_verify_preparation_lifetime_v1";')
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
(R/'source-bindings1.json').write_text(json.dumps(dict(base_source=str(B),files_sha256=files,changed_sha256={n:files[n] for n in sorted(changed)},default_option_on=False,no_xmodel_operator_change=True,goal_complete=False),indent=2)+'\n')
print(json.dumps(dict(files=len(files),changed=sorted(changed))))
