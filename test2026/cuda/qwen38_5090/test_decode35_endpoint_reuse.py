import copy,hashlib,json,unittest
from .decode35_endpoint_reuse import checked_reuse,J,P,B

class MemoryZip:
    def __init__(self,data):self.data=data
    def read(self,key):return self.data[key]

def fixture():
    old={k:{} for k in ('runtime','environment','checkpoints','flags','historical_libraries')}
    old.update(cuda_objects={str(i):'object' for i in range(34)},source_files={str(i):'source' for i in range(86)},candidate_base={str(i):None for i in range(86)},files={'original':'hash'},installed_files={'original':'hash','source86':'hash'},cache_before_sha256='cache',input_corpus_sha256='corpus',selected_key='selected',bundle='source86.zip',bundle_sha256='bundle',library='candidate.dll',library_sha256='a'*64)
    q=dict(status='same_DLL_decode35_operands_failure_replayed',exit_code=1,analysis=None,error="AssertionError('Observer or DLL changed complete primary state')",CPU_controls=81,source_runtime_SDK_config_index_CMake_all34_baseline_restored=True,controls={p:{'passed':True} for p in ('off-reference','off-candidate','on-reference')},archive_sha256='raw')
    p=copy.deepcopy(old);p['files']['new_method']='new';p['parents']={J:{'archive_sha256':'raw'}}
    raw=json.dumps(old).encode();w=dict(preparation_sha256=hashlib.sha256(raw).hexdigest(),cleanup_finished=True,all_restored=True,exit_code=1,source_before=old['files'],source_after=old['files'],cuda_objects_before=old['cuda_objects'],cuda_objects_after=old['cuda_objects'],phases=[{'phase':phase,'exit_code':0,'source_installed':old['installed_files'],'library_sha256':'a'*64} for phase in ('off-reference','off-candidate','on-reference','on-candidate')])
    data={P+'evidence/decode35-operands-preparation.json':raw,B+'workflow.json':json.dumps(w).encode()}
    for phase,batch in (('off-reference',1),('off-candidate',4)):
        data[B+phase+'/report.json']=json.dumps(dict(phase=phase,mode=phase.split('-')[1],physical_batch=batch,operand_witness=False,internal_observer=None,rows=[{'operand_observations':[]}],library_sha256='a'*64)).encode();data[B+phase+'/blob']=phase.encode()
    return p,MemoryZip(data),q

class Reuse(unittest.TestCase):
    def test_closed_failure_can_supply_only_verified_ordinary_profiles(self):
        p,z,q=fixture();reports,reads,record=checked_reuse(p,z,q)
        self.assertEqual(set(reports),{'off-reference','off-candidate'})
        self.assertEqual(reads['off-reference']('blob'),b'off-reference');self.assertEqual(reads['off-candidate']('blob'),b'off-candidate')
        self.assertTrue(record['all_new_on_primary_controls_required']);self.assertTrue(all(not r['fresh_process_in_this_experiment'] for r in record['profiles'].values()))

    def test_identity_drift_in_flags_weights_DLL_objects_or_old_source_rejected(self):
        for name in ('flags','checkpoints','library_sha256','cuda_objects','files','input_corpus_sha256'):
            p,z,q=fixture()
            if name=='library_sha256':p[name]='b'*64
            elif name=='input_corpus_sha256':p[name]='other'
            else:p[name]['original' if name=='files' else 'changed']='bad'
            with self.subTest(field=name),self.assertRaises(AssertionError):checked_reuse(p,z,q)

    def test_incomplete_or_unrestored_or_unvalidated_old_control_rejected(self):
        for name in ('phase_exit','restoration','control','observer'):
            p,z,q=fixture()
            if name in ('phase_exit','restoration'):
                w=json.loads(z.read(B+'workflow.json'))
                if name=='phase_exit':w['phases'][0]['exit_code']=1
                else:w['all_restored']=False
                z.data[B+'workflow.json']=json.dumps(w).encode()
            elif name=='control':q['controls'].pop('off-candidate')
            else:
                r=json.loads(z.read(B+'off-candidate/report.json'));r['operand_witness']=True;z.data[B+'off-candidate/report.json']=json.dumps(r).encode()
            with self.subTest(field=name),self.assertRaises(AssertionError):checked_reuse(p,z,q)

if __name__=='__main__':unittest.main()
