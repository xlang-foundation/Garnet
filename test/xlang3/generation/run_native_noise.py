# SPDX-License-Identifier: Apache-2.0
"""Bulk native generation noise must preserve the existing seed contract."""
import random,tensor,garnet,time
for seed in (0,1,123,2**32-1,2**32+7,2**63-1,2**63,2**64-1):
    for count in (1,2,3,7,10001):
        generator=random.Random(seed)
        expected=tensor.tensor([generator.gauss(0,1) for _ in range(count)],dtype=tensor.float32).tolist()
        actual=garnet.gaussian_noise([count],str(seed)).tolist()
        assert actual==expected,(seed,count,actual[:3],expected[:3])
for shape,seed in (([],0),([0],0),([-1],0),([True],0),([2**62],0),([2],-1),([2],2**64)):
    failed=False
    try:garnet.gaussian_noise(shape,str(seed))
    except Exception:failed=True
    assert failed,(shape,seed)
started=time.perf_counter();result=garnet.gaussian_noise([1,128,16,22,40],'0')
assert list(result.shape)==[1,128,16,22,40]
print('Native Gaussian noise passed exact float32 parity across 40 seed/size cases; large profile seconds',time.perf_counter()-started,flush=True)
