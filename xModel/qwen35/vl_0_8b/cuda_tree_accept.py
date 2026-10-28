# SPDX-License-Identifier: Apache-2.0
"""Prepared host greedy tree acceptance; not installed in a serving route.

Every accepted draft equals its verified parent's target prediction. The bonus
is carried to the next iteration; the caller must not emit it a second time.
EOS and the final output token are sampled without processing those inputs.
"""

def select_greedy_paths(proposal,predictions,working,remaining,stops):
    tokens=proposal['tokens'];parents=proposal['parents'];active=proposal['active'];used=proposal['used_nodes']
    b=len(tokens)
    if b not in (1,2,4,8) or not tokens:raise ValueError('Bounded batch required')
    n=len(tokens[0])
    if n not in (8,16,32,64) or len(predictions)!=b*n or not all(len(v)==b for v in (parents,active,used,working,remaining)):
        raise ValueError('Tree acceptance shape mismatch')
    emitted=[];sampled=[];next_tokens=[];counts=[];selected=[];paths=[];keep_working=[];reasons=[];stop_ids=[];accepted=[]
    for r in range(b):
        if len(tokens[r])!=n or len(parents[r])!=n or len(active[r])!=n or type(used[r]) is not int or not 1<=used[r]<=n:
            raise ValueError('Tree row dimensions invalid')
        if working[r] not in (0,1) or type(remaining[r]) is not int or remaining[r]<1:raise ValueError('Invalid output controls')
        children={}
        for node in range(used[r]):
            if type(tokens[r][node]) is not int or tokens[r][node]<0 or active[r][node] not in (0,1):raise ValueError('Invalid token/active controls')
            p=parents[r][node]
            if type(p) is not int or (p!=-1 if node==0 else not 0<=p<node):raise ValueError('Invalid parent')
            if node:
                key=(p,tokens[r][node])
                if key in children:raise ValueError('Duplicate sibling token creates ambiguous selection')
                children[key]=node
        out=[];sample=0;path=[];next_token=tokens[r][0];ongoing=bool(working[r]);reason=None;stop=None;draft_count=0
        if ongoing:
            sample=1
            if next_token in stops:ongoing=False;reason='eos';stop=next_token
            else:
                out.append(next_token)
                if remaining[r]==1:ongoing=False;reason='length'
                else:
                    node=0
                    while True:
                        if not active[r][node]:raise ValueError('An accepted nonterminal input is inactive')
                        path.append(node)
                        if len(path)>8:raise ValueError('Accepted path exceeds native bound')
                        actual=predictions[r*n+node]
                        if type(actual) is not int or actual<0:raise ValueError('Invalid verified target token')
                        child=children.get((node,actual))
                        next_token=actual
                        if child is None:break
                        draft_count+=1;sample+=1
                        if actual in stops:ongoing=False;reason='eos';stop=actual;break
                        out.append(actual)
                        if sample==remaining[r]:ongoing=False;reason='length';break
                        node=child
        emitted.append(out);sampled.append(sample);next_tokens.append(next_token);counts.append(len(path));selected.append(path[-1] if path else -1)
        paths.append(path);keep_working.append(int(ongoing));reasons.append(reason);stop_ids.append(stop);accepted.append(draft_count)
    return {'emitted_token_ids':emitted,'sampled_counts':sampled,'next_tokens':next_tokens,'accepted_input_counts':counts,
        'selected_last_nodes':selected,'accepted_paths':paths,'working':keep_working,'finish_reasons':reasons,
        'stop_token_ids':stop_ids,'accepted_draft_tokens':accepted}
