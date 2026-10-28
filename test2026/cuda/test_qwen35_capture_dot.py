"""Positive and negative controls for node-type eligibility and full parsing."""
import json
from pathlib import Path
import sys
from qwen35_capture_dot import parse_dot
root=Path(sys.argv[1]).resolve();output=root/'qwen35-capture-topology-v1-parser-controls.json'
assert not output.exists()
def graph(labels,edges=()):
    return 'digraph dot {\n'+''.join(f'"graph_1_node_{i}"[label="{{{label}\n| {{ID | {i}}}}}"];\n' for i,label in enumerate(labels))+''.join(f'"graph_1_node_{a}" -> "graph_1_node_{b}";\n' for a,b in edges)+'}\n'
assert parse_dot(graph(['KERNEL','EMPTY'],[(0,1)]))['all_nodes_kernel_or_empty']
assert not parse_dot(graph(['KERNEL','HOST'],[(0,1)]))['all_nodes_kernel_or_empty']
assert not parse_dot(graph(['MEM_ALLOC']))['all_nodes_kernel_or_empty']
assert not parse_dot(graph(['MEMCPY']))['all_nodes_kernel_or_empty']
bad=[graph(['KERNEL'],[(0,1)]),graph(['KERNEL','KERNEL'],[(0,1),(1,0)]),
     graph(['KERNEL']).replace('];',''),graph(['KERNEL']).replace('KERNEL','not_a_type'),
     graph(['KERNEL'])[:-3],graph(['KERNEL']).replace('}\n','"graph_1_node_0"[label="{KERNEL\n}"];\n}\n')]
for source in bad:
    try:parse_dot(source)
    except AssertionError:pass
    else:raise AssertionError('Malformed/partial/cyclic DOT admitted')
actual=root/'capture-garnet-cuda-graph-71355333737800-2.dot'
parsed=parse_dot(actual.read_text())
assert parsed['node_count']==52 and parsed['all_node_types_permitted_in_conditional_body']
assert not parsed['all_nodes_kernel_or_empty'] and parsed['node_type_counts']['MEMCPY']==3
with output.open('x') as stream:json.dump({'all_controls_passed':True,'synthetic_controls':10,'historical_format_fixture':str(actual),'historical_format_result':parsed,'scope':'CPU parser coverage only; historical DOT is not current model evidence.'},stream,indent=2)
print('CAPTURE_DOT_PARSER_CONTROLS_PASS')
