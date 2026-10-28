"""Five explicit padding/order/repeat cases from the eight pinned 4K inputs."""
import hashlib
def select_lifecycle(inputs):
    assert type(inputs) is dict and len(inputs)==8
    text=[];image=[]
    for key,row in inputs.items():
        assert len(row['prompt_ids'])==3968 and row['body']['ignore_eos'] is True
        content=row['body']['messages'][0]['content']
        assert isinstance(content,str) or type(content) is list and len(content)==2
        (text if isinstance(content,str) else image).append(key)
    assert len(text)==len(image)==4
    specs=[('padding-one-text',[text[0]]),('padding-two-image',[image[0],image[3]]),
           ('padding-three-mixed',[text[2],image[1],text[1]]),
           ('permuted-four-mixed',[image[2],text[3],image[0],text[0]]),
           ('repeat-three-mixed',[text[2],image[1],text[1]])]
    groups=[dict(label=label,real_rows=len(keys),real_keys=list(keys),keys=keys+[keys[0]]*(4-len(keys)),
                 public_ids=[hashlib.sha256(('B4-padding-lifecycle:'+label+':'+str(i)).encode()).hexdigest()[:32] for i in range(len(keys))],
                 generated_IDs=128,total_context_capacity=4096) for label,keys in specs]
    return dict(status='retained_fixed4k_B4_padding_lifecycle_inputs_selected_not_executed',inputs=inputs,groups=groups,
                physical_generated_ID_values=2560,public_generated_ID_values=1664,full_phase_row_comparisons=80,
                quality_HTTP_or_performance_qualified=False)
