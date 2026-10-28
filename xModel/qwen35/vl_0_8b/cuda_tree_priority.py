# SPDX-License-Identifier: Apache-2.0
"""Proposal-only costs conditioned on two logits, not full-vocabulary probabilities."""
import math

def binary_top2_costs(scores):
    if len(scores)!=2 or any(type(value) not in (int,float) or not math.isfinite(value) for value in scores):
        raise ValueError('Two finite numeric proposal scores required')
    if scores[1]>scores[0]:
        raise ValueError('Proposal scores must be descending')
    delta=float(scores[1])-float(scores[0])
    first=math.log1p(math.exp(delta))
    second=first-delta
    if not all(math.isfinite(value) and value>=0 for value in (first,second)):
        raise ValueError('Invalid binary-conditioned path costs')
    return [first,second]

def anchored_cost_options(options,spine_nodes,limit):
    """Preserve the greedy rank-0 chain, spend remaining slots on lowest costs."""
    if type(limit) is not int or limit<0:
        raise ValueError('A nonnegative integer frontier limit is required')
    ordered=sorted(options)
    anchors=[value for value in ordered if value[2] in spine_nodes and value[1]%2==0]
    if len(anchors)>1:
        raise ValueError('A frontier may contain at most one greedy-spine edge')
    if not anchors or not limit:
        return ordered[:limit]
    anchor=anchors[0]
    return [anchor]+[value for value in ordered if value!=anchor][:limit-1]
