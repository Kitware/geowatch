import ubelt as ub
import networkx as nx
import parso
from liberator.core import undefined_names

fpath = ub.Path('~/code/watch/watch/cli/run_tracker.py').expand()
module = parso.parse(fpath.read_text())

funcs = list(module.iter_funcdefs())
func_names = {f.name.value for f in funcs}

graph = nx.DiGraph()

for func in funcs:
    func_name = func.name.value
    freevars = undefined_names(func.get_code())
    referenced_freevars = set(freevars) & func_names
    print(f'referenced_freevars={referenced_freevars}')

    graph.add_node(func_name)
    for varname in referenced_freevars:
        graph.add_edge(func_name, varname)

nx.write_network_text(graph)
