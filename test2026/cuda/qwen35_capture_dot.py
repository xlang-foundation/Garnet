"""Strict reader for CUDA verbose DOT node/edge declarations; fail closed."""
import collections
import re

def parse_dot(source):
    assert source.lstrip().startswith('digraph '),'Not a directed CUDA DOT'
    declarations=list(re.finditer(r'(?m)^"([^"\r\n]+)"\s*\[',source))
    assert declarations,'No node declarations'
    nodes={}
    for match in declarations:
        name=match.group(1);assert name not in nodes,'Duplicate node declaration'
        begin=match.end()-1;depth=0;quoted=False;escaped=False;end=None
        for i in range(begin,len(source)):
            char=source[i]
            if quoted:
                if escaped:escaped=False
                elif char=='\\':escaped=True
                elif char=='"':quoted=False
            elif char=='"':quoted=True
            elif char=='[':depth+=1
            elif char==']':
                depth-=1
                if depth==0:end=i;break
        assert end is not None and not quoted,'Truncated node attributes'
        attrs=source[begin+1:end]
        label=re.search(r'\blabel\s*=\s*"\{\s*([A-Z][A-Z0-9_ -]*)(?:\r?\n|\||\})',attrs)
        assert label,('Unknown CUDA DOT node label format',name,attrs[:160])
        nodes[name]={'type':label.group(1),'attributes':attrs}
    edges=re.findall(r'(?m)^"([^"\r\n]+)"\s*->\s*"([^"\r\n]+)"[^\r\n]*;',source)
    assert source.rstrip().endswith('}'),'Truncated DOT graph'
    successors={name:[] for name in nodes};incoming={name:0 for name in nodes}
    assert len(edges)==len(set(edges)),'Duplicate edge'
    for left,right in edges:
        assert left in nodes and right in nodes,'Edge references missing node'
        successors[left].append(right);incoming[right]+=1
    ready=[name for name in nodes if not incoming[name]];visited=0
    while ready:
        name=ready.pop();visited+=1
        for target in successors[name]:
            incoming[target]-=1
            if not incoming[target]:ready.append(target)
    assert visited==len(nodes),'DOT graph has a cycle'
    counts=dict(collections.Counter(x['type'] for x in nodes.values()))
    return {'node_count':len(nodes),'edge_count':len(edges),'node_type_counts':counts,
            'all_nodes_kernel_or_empty':all(x in ('KERNEL','EMPTY') for x in counts),
            'all_node_types_permitted_in_conditional_body':all(x in ('KERNEL','EMPTY','MEMCPY','MEMSET','GRAPH','CONDITIONAL') for x in counts),
            'memory_nodes_requiring_operand_accessibility_checks':sum(v for k,v in counts.items() if k in ('MEMCPY','MEMSET')),
            'memcpy_labels':[x['attributes'] for x in nodes.values() if x['type']=='MEMCPY'],
            'scope':'Node-type/topology screen only. Does not prove current device/context ownership, kernel absence of device launch, operand accessibility or successful conditional-body instantiation.'}
