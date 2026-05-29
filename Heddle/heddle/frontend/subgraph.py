import torch.fx as fx
from torch.fx.node import map_arg
from typing import Dict, Iterable, List, Set, Tuple


def extract_subgraph_gm(
    gm: fx.GraphModule,
    nodes: Set[fx.Node],
    input_nodes: Set[fx.Node],
    output_nodes: Set[fx.Node],
) -> fx.GraphModule:
    """
    Build a new GraphModule that contains only `nodes` and the required placeholders for `input_nodes`.
    The output is a single-tensor output if `output_nodes` has size 1; otherwise returns a tuple.
    """
    graph = fx.Graph()
    env: Dict[fx.Node, fx.Node] = {}

    # Deterministic ordering: respect original graph order
    orig_nodes: List[fx.Node] = list(gm.graph.nodes)

    # Create placeholders for external inputs (may include original placeholders or intermediate nodes)
    for n in orig_nodes:
        if n in input_nodes:
            ph = graph.placeholder(n.name)
            # Preserve meta (dtype/shape) if present so codegen can infer buffer dtypes.
            try:
                ph.meta = dict(getattr(n, "meta", {}) or {})
            except Exception:
                pass
            env[n] = ph

    # Copy internal nodes
    for n in orig_nodes:
        if n not in nodes:
            continue
        if n.op not in ("call_function", "call_method"):
            # Skip things we don't handle in this prototype
            continue

        def remap(a):
            if isinstance(a, fx.Node):
                if a in env:
                    return env[a]
                # If this is an internal dependency not already materialized, treat it as a placeholder.
                # (This can happen if Partition.inputs was incomplete.)
                ph = graph.placeholder(a.name)
                try:
                    ph.meta = dict(getattr(a, "meta", {}) or {})
                except Exception:
                    pass
                env[a] = ph
                return ph
            return a

        new_args = map_arg(n.args, remap)
        new_kwargs = map_arg(n.kwargs, remap)

        if n.op == "call_function":
            new_n = graph.call_function(n.target, new_args, new_kwargs)
            try:
                new_n.meta = dict(getattr(n, "meta", {}) or {})
            except Exception:
                pass
            env[n] = new_n
        else:
            # call_method: first arg is self
            new_n = graph.call_method(n.target, new_args, new_kwargs)
            try:
                new_n.meta = dict(getattr(n, "meta", {}) or {})
            except Exception:
                pass
            env[n] = new_n

    # Output
    outs: List[fx.Node] = []
    for n in orig_nodes:
        if n in output_nodes and n in env:
            outs.append(env[n])

    if len(outs) == 0:
        # Fallback: last internal node in topo order
        for n in reversed(orig_nodes):
            if n in nodes and n in env:
                outs = [env[n]]
                break

    if len(outs) == 1:
        graph.output(outs[0])
    else:
        graph.output(tuple(outs))

    sub_gm = fx.GraphModule(gm, graph)
    sub_gm.recompile()
    return sub_gm


