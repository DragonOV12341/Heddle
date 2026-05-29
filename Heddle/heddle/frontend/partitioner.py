# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.fx as fx
from typing import List, Set, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import operator
from collections import deque
import os
from .subgraph import extract_subgraph_gm
from .codegen import ElementwiseCodegen
from tilelang.tools import Analyzer
from tilelang.carver.arch.cuda import CUDA
from .support import SupportPolicy

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return default

def _target_key(target: Any) -> str:
    """
    Try to get a stable-ish textual key for FX node targets.
    This matters because torch.compile graphs may use `torch.ops.aten.*` targets.
    """
    if target is None:
        return "None"
    try:
        if hasattr(target, "__name__"):
            return str(target.__name__)
    except Exception:
        pass
    try:
        # torch._ops.OpOverload has .name() in some versions
        if hasattr(target, "name") and callable(getattr(target, "name")):
            return str(target.name())
    except Exception:
        pass
    try:
        return str(target)
    except Exception:
        return repr(target)

def _dtype_nbytes(dtype: Optional[torch.dtype]) -> int:
    if dtype is None:
        return 0
    if dtype == torch.float16:
        return 2
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == torch.float64:
        return 8
    # conservative fallback
    return 4

def _node_tensor_meta(node: fx.Node):
    try:
        return node.meta.get("tensor_meta", None)
    except Exception:
        return None

def _node_numel(node: fx.Node) -> Optional[int]:
    """
    Best-effort numel from ShapeProp's tensor_meta (symbolic shapes return None).
    """
    tm = _node_tensor_meta(node)
    if tm is None or not hasattr(tm, "shape"):
        return None
    try:
        n = 1
        for d in tm.shape:
            if isinstance(d, int):
                n *= int(d)
            else:
                # SymInt / symbolic
                return None
        return int(n)
    except Exception:
        return None

def _node_dtype(node: fx.Node) -> Optional[torch.dtype]:
    tm = _node_tensor_meta(node)
    if tm is not None and hasattr(tm, "dtype"):
        try:
            return tm.dtype
        except Exception:
            pass
    try:
        v = node.meta.get("val", None)
        if isinstance(v, torch.Tensor):
            return v.dtype
    except Exception:
        pass
    return None

@dataclass
class LoopAnalysisResult:
    # Set of parallel axes (indices)
    parallel_axes: Set[int] = field(default_factory=set)
    # Set of reduction axes (indices)
    reduction_axes: Set[int] = field(default_factory=set)
    # Rank of the output tensor
    rank: int = 0

class LoopAnalyzer:
    """
    Analyzes the loop structure of a partition based on FakeTensor shapes and op semantics.
    """
    def __init__(self):
        pass

    def analyze(self, nodes: Set[torch.fx.Node]) -> Dict[torch.fx.Node, LoopAnalysisResult]:
        results = {}
        for node in nodes:
            results[node] = self._analyze_node(node)
        return results

    def _analyze_node(self, node: torch.fx.Node) -> LoopAnalysisResult:
        # Default: Elementwise (parallel on all axes)
        rank = 0
        if 'val' in node.meta:
            val = node.meta['val']
            if isinstance(val, (torch.Tensor, torch.SymInt)): # FakeTensor or SymInt
                if hasattr(val, 'ndim'): rank = val.ndim
                elif hasattr(val, 'dim'): rank = val.dim() # SymInt might not have dim
        
        # If no meta, try to guess from args (heuristic)
        if rank == 0 and len(node.args) > 0 and hasattr(node.args[0], 'meta'):
             # Inherit rank from input
             pass

        res = LoopAnalysisResult(rank=rank)
        
        # Assume all axes are parallel unless specified otherwise
        # This is true for elementwise ops
        res.parallel_axes = set(range(rank))

        if node.op == 'call_function':
            if node.target in [torch.sum, torch.mean, torch.max, torch.min, torch.prod]:
                # Reduction Op
                dim = None
                if len(node.args) > 1:
                    dim = node.args[1]
                else:
                    dim = node.kwargs.get('dim')
                
                keepdim = node.kwargs.get('keepdim', False)
                
                if dim is not None:
                    if isinstance(dim, int): dims = (dim,)
                    else: dims = dim
                    
                    # Resolve negative dims
                    resolved_dims = set()
                    input_rank = rank # Approx, ideally use input's rank
                    # Try to get input rank
                    if len(node.args) > 0 and hasattr(node.args[0], 'meta'):
                         val = node.args[0].meta.get('val')
                         if hasattr(val, 'ndim'): input_rank = val.ndim

                    for d in dims:
                        if d < 0: d += input_rank
                        resolved_dims.add(d)
                    
                    res.reduction_axes = resolved_dims
                    # Reduced axes are NOT parallel in the OUTPUT domain
                    # But wait, LoopAnalysisResult describes the Operation's Iteration Domain?
                    # Or the Output Tensor's domain?
                    # Let's define it as "Axes of the Input Iteration Space used by this Op"
                    
                    # For a Reduction Op:
                    # It iterates over Input Domain.
                    # Some axes are Parallel (batch), some are Reduction.
                    # We mark reduction axes explicitly.
                    pass
            
            elif node.target in [torch.matmul, torch.mm, torch.bmm]:
                # MatMul: (..., M, K) x (..., K, N) -> (..., M, N)
                # Parallel axes: Batch (B), M, N
                # Reduction axis: K
                # This is complex because 'dim' is implicit.
                # We need to map Input axes to Output axes.
                # For H2O check, simpler heuristic: MatMul implies a reduction on the contracting dim.
                # Last dim of LHS and Second-to-last of RHS.
                pass

        return res

    def check_compatibility(self, partition_nodes: Set[torch.fx.Node]) -> bool:
        """
        Check if all nodes in the partition are compatible for fusion.
        Key check: Conflicting reduction axes.
        """
        results = self.analyze(partition_nodes)
        
        # Collect all reduction axes across the partition
        # We need to be careful: 
        # Reduction on Axis 2 in Op A might be totally unrelated to Axis 2 in Op B if there's a Transpose in between.
        # BUT, if we assume no complex Reshape/Permute (or pruning handles them), 
        # checking absolute axes indices is a reasonable first-order approximation.
        
        seen_reduction_axes = set()
        
        for node, res in results.items():
            if res.reduction_axes:
                # If we see multiple different reduction axes in one kernel, it's a strong signal of conflict
                # e.g. Sum(dim=2) and Sum(dim=3) -> usually means Row-wise AND Col-wise reduction.
                # Hard to fuse efficiently without global atomic sync.
                
                # Check intersection with already seen axes
                # If disjoint and both non-empty -> Conflict
                if seen_reduction_axes and seen_reduction_axes.isdisjoint(res.reduction_axes):
                     # Conflict: Reducing on different dimensions
                     return False
                
                seen_reduction_axes.update(res.reduction_axes)
        
        return True

@dataclass
class Partition:
    nodes: Set[torch.fx.Node] = field(default_factory=set)
    inputs: Set[torch.fx.Node] = field(default_factory=set)
    outputs: Set[torch.fx.Node] = field(default_factory=set)
    score: float = 0.0 # Could be latency or heuristic score

    def add_node(self, node: torch.fx.Node):
        self.nodes.add(node)

    def is_connected(self) -> bool:
        return True

    def calculate_arithmetic_intensity(self) -> float:
        """
        Best-effort AI estimation: total flops / (bytes of fused IO).
        This is intentionally conservative; we use it mainly for pruning obviously-bad candidates.
        """
        flops = 0.0
        # For elementwise chains, use output numel as the main "work unit".
        # For reductions, use input numel.
        for node in self.nodes:
            if node.op not in ("call_function", "call_method"):
                continue
            # Estimate per-element cost. (These weights are crude but stable.)
            per_elem = 1.0
            k = _target_key(getattr(node, "target", None))
            if node.op == "call_function":
                if node.target in (torch.exp, torch.sigmoid) or "aten.exp" in k or "aten.sigmoid" in k:
                    per_elem = 10.0
            else:
                if str(node.target) in ("exp", "sigmoid"):
                    per_elem = 10.0

            # Reduction counts roughly as one op per input element.
            is_reduction = False
            if node.op == "call_function":
                is_reduction = (node.target in (torch.sum, torch.mean, torch.amax)) or any(x in k for x in ("aten.sum", "aten.mean", "aten.amax"))
            else:
                is_reduction = str(node.target) in ("sum", "mean", "amax")

            n_elem = None
            if is_reduction and len(node.args) > 0 and isinstance(node.args[0], fx.Node):
                n_elem = _node_numel(node.args[0])
            if n_elem is None:
                n_elem = _node_numel(node)
            if n_elem is None:
                # Symbolic shape: fall back to node count weighting
                n_elem = 1
            flops += per_elem * float(n_elem)

        bytes_accessed = 0.0
        for v in list(self.inputs) + list(self.outputs):
            n_elem = _node_numel(v)
            dt = _node_dtype(v)
            if n_elem is None:
                continue
            bytes_accessed += float(n_elem) * float(_dtype_nbytes(dt))

        return float(flops / (bytes_accessed + 1e-9))

    def estimate_work_items(self) -> Optional[int]:
        """
        Heuristic proxy for parallelism: prefer large partitions with enough elements/rows.
        """
        # Prefer output node's numel (most relevant for elementwise).
        best = None
        for o in self.outputs:
            n = _node_numel(o)
            if n is not None:
                best = max(best or 0, n)
        if best is not None:
            return int(best)
        # Fallback to any node's numel
        for n in self.nodes:
            nn = _node_numel(n)
            if nn is not None:
                best = max(best or 0, nn)
        return int(best) if best is not None else None

    def check_loop_compatibility(self) -> bool:
        """
        Delegates to LoopAnalyzer for advanced compatibility checks.
        """
        analyzer = LoopAnalyzer()
        return analyzer.check_compatibility(self.nodes)

class GraphPartitioner:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Optional[List[torch.Tensor]] = None,
        support_policy: Optional[SupportPolicy] = None,
    ):
        self.gm = gm
        self.example_inputs = example_inputs or []
        self.support = support_policy or SupportPolicy.default()
        self.nodes = list(gm.graph.nodes)
        self.candidates: List[Partition] = []
        self._analyzer_cache: Dict[Tuple[int, ...], float] = {}
        # Graph outputs (nodes that directly feed the output tuple)
        self.graph_output_nodes: Set[torch.fx.Node] = set()
        for n in gm.graph.nodes:
            if n.op == "output":
                out = n.args[0]
                if isinstance(out, tuple):
                    for o in out:
                        if isinstance(o, torch.fx.Node):
                            self.graph_output_nodes.add(o)
                elif isinstance(out, torch.fx.Node):
                    self.graph_output_nodes.add(out)
                break

    def partition(self, cost_model: str = "heuristic") -> List[Partition]:
        self.support.debug_dump(self.gm)
        print("[Partitioner] Phase 1: Enumerating subgraphs...")
        enum_strategy = os.environ.get("TILELANG_FRONTEND_ENUM_STRATEGY", "auto").strip().lower()
        max_candidates = _env_int("TILELANG_FRONTEND_ENUM_MAX_CANDIDATES", 2000)
        max_depth = _env_int("TILELANG_FRONTEND_ENUM_MAX_DEPTH", 6)  # for grow
        max_nodes = _env_int("TILELANG_FRONTEND_ENUM_MAX_NODES", 10)  # for connected DFS
        auto_threshold = _env_int("TILELANG_FRONTEND_ENUM_AUTO_THRESHOLD", 80)

        # Decide enumeration strategy.
        eligible = self._eligible_compute_nodes()
        print(f"[Partitioner] Eligible compute nodes: {len(eligible)}")
        if enum_strategy == "auto":
            use_connected = len(eligible) <= auto_threshold
        elif enum_strategy in ("connected", "connected_dfs", "dfs"):
            use_connected = True
        else:
            use_connected = False

        if use_connected:
            print(f"[Partitioner] Enum strategy: connected_dfs (max_nodes={max_nodes}, max_candidates={max_candidates})")
            self._enumerate_connected_dfs(max_nodes=max_nodes, max_candidates=max_candidates)
        else:
            print(f"[Partitioner] Enum strategy: grow (max_depth={max_depth}, max_candidates={max_candidates})")
            self._enumerate_subgraphs(max_depth=max_depth, max_candidates=max_candidates)
        
        print(f"[Partitioner] Found {len(self.candidates)} candidates.")
        
        print("[Partitioner] Phase 2: Pruning...")
        self._prune_candidates()
        print(f"[Partitioner] {len(self.candidates)} candidates remained.")
        
        print("[Partitioner] Phase 3: Selection (Greedy)...")
        if not self.candidates:
            return []

        if cost_model == "analyzer":
            self._score_candidates_with_analyzer()
            # Higher is better
            self.candidates.sort(key=lambda p: p.score, reverse=True)
        else:
            # Heuristic scoring: prefer high AI and enough work to amortize launch/materialization.
            for p in self.candidates:
                ai = p.calculate_arithmetic_intensity()
                work = p.estimate_work_items() or 1
                # score combines compute density and size (very rough)
                p.score = float(ai) * float(work) * float(len(p.nodes) ** 1.5)
            self.candidates.sort(key=lambda p: p.score, reverse=True)
        
        selected = []
        covered_nodes = set()
        
        for cand in self.candidates:
            if not cand.nodes.intersection(covered_nodes):
                selected.append(cand)
                covered_nodes.update(cand.nodes)
                
        return selected

    def _eligible_compute_nodes(self) -> List[fx.Node]:
        """
        Nodes that are allowed to be inside a candidate partition.
        """
        eligible: List[fx.Node] = []
        for n in self.nodes:
            if n.op not in ("call_function", "call_method"):
                continue
            if self.support.is_any_boundary(n):
                continue
            if not self.support.is_supported_inside_partition(n):
                continue
            eligible.append(n)
        return eligible

    def _build_undirected_adjacency(self, eligible: List[fx.Node]) -> Tuple[Dict[int, Set[int]], Dict[fx.Node, int], Dict[int, fx.Node]]:
        """
        Build an undirected adjacency graph among eligible compute nodes, using dataflow edges (args/users).
        """
        node2id: Dict[fx.Node, int] = {}
        id2node: Dict[int, fx.Node] = {}
        for i, n in enumerate(eligible):
            node2id[n] = i
            id2node[i] = n

        adj: Dict[int, Set[int]] = {i: set() for i in id2node.keys()}

        def _add_edge(a: fx.Node, b: fx.Node):
            if a not in node2id or b not in node2id:
                return
            ia = node2id[a]
            ib = node2id[b]
            if ia == ib:
                return
            adj[ia].add(ib)
            adj[ib].add(ia)

        for n in eligible:
            # args -> n
            for a in n.args:
                if isinstance(a, fx.Node) and a in node2id:
                    _add_edge(n, a)
            # n -> users
            for u in n.users:
                if u in node2id:
                    _add_edge(n, u)

        return adj, node2id, id2node

    def _enumerate_connected_dfs(self, max_nodes: int, max_candidates: int):
        """
        Enumerate connected subgraphs via DFS (ahlo-style).

        This is strictly stronger than the current grow/greedy enumeration because it can expand
        in both directions and explore multiple frontiers, not just follow a single chain.
        """
        eligible = self._eligible_compute_nodes()
        if not eligible:
            return

        # Deterministic ordering: use original topo order of self.nodes.
        topo_rank = {n: i for i, n in enumerate(self.nodes)}
        eligible.sort(key=lambda n: topo_rank.get(n, 1 << 30))

        adj, node2id, id2node = self._build_undirected_adjacency(eligible)

        # Dedup by id-set
        seen: Set[frozenset[int]] = set()
        excluded: Set[int] = set()

        # Helper to materialize a candidate into Partition
        def _emit(cur: frozenset[int]):
            if len(self.candidates) >= max_candidates:
                return
            # Skip trivial size-1 candidates; pruning will also remove them but this saves budget.
            if len(cur) < 2:
                return
            nodes = {id2node[i] for i in cur}
            self._finalize_candidate(Partition(nodes=nodes))

        # DFS
        def dfs(cur: frozenset[int], neighbors: Set[int], excluded_local: Set[int]):
            if len(self.candidates) >= max_candidates:
                return
            _emit(cur)
            if len(cur) >= max_nodes:
                return
            # Only expand to nodes not in excluded_local
            cand_next = neighbors - excluded_local
            if not cand_next:
                return
            # Deterministic: increasing id order
            for nid in sorted(cand_next):
                if len(self.candidates) >= max_candidates:
                    return
                new_cur = frozenset(set(cur) | {nid})
                if new_cur in seen:
                    continue
                seen.add(new_cur)
                new_neighbors = set(neighbors)
                new_neighbors.update(adj.get(nid, set()))
                dfs(new_cur, new_neighbors, set(excluded_local) | {nid})

        for n in eligible:
            if len(self.candidates) >= max_candidates:
                break
            nid = node2id[n]
            excluded.add(nid)
            start = frozenset({nid})
            # Note: we don't add start to seen because we skip size-1 emits.
            dfs(start, set(adj.get(nid, set())), set(excluded))

    def _enumerate_subgraphs(self, max_depth: int, max_candidates: int):
        seen: Set[frozenset] = set()
        for root in self.nodes:
            if root.op not in ('call_function', 'call_method'):
                continue
            if self.support.is_any_boundary(root) or (not self.support.is_supported_inside_partition(root)):
                continue
            current_p = Partition()
            current_p.nodes = {root}
            self._grow_partition(current_p, root, depth=0, max_depth=max_depth, seen=seen, max_candidates=max_candidates)
            if len(self.candidates) >= max_candidates:
                break

    def _grow_partition(
        self,
        partition: Partition,
        current_node: torch.fx.Node,
        depth: int,
        max_depth: int,
        seen: Set[frozenset],
        max_candidates: int,
    ):
        if depth >= max_depth:
            self._finalize_candidate(partition)
            return

        # Undirected neighbors: users + args (keeps connectivity while exploring more than a forward chain).
        neighbors: List[fx.Node] = []
        try:
            neighbors.extend([n for n in current_node.users])
        except Exception:
            pass
        try:
            for a in current_node.args:
                if isinstance(a, fx.Node):
                    neighbors.append(a)
        except Exception:
            pass

        # Filter neighbors to supported compute nodes only.
        filtered = []
        for n in neighbors:
            if n in partition.nodes:
                continue
            if n.op not in ("call_function", "call_method"):
                continue
            if self.support.is_any_boundary(n) or (not self.support.is_supported_inside_partition(n)):
                continue
            filtered.append(n)
        
        if not filtered:
            self._finalize_candidate(partition)
            return

        self._finalize_candidate(partition)
        
        # Deterministic: follow original topo order (self.nodes).
        order = {n: i for i, n in enumerate(self.nodes)}
        filtered.sort(key=lambda n: order.get(n, 1 << 30))
        for neighbor in filtered:
            if len(self.candidates) >= max_candidates:
                return
            new_nodes = frozenset(partition.nodes | {neighbor})
            if new_nodes in seen:
                continue
            seen.add(new_nodes)
            new_p = Partition(nodes=set(new_nodes))
            self._grow_partition(new_p, neighbor, depth + 1, max_depth, seen, max_candidates)

    def _finalize_candidate(self, partition: Partition):
        all_nodes = partition.nodes
        inputs = set()
        outputs = set()
        
        for node in all_nodes:
            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and arg not in all_nodes:
                    inputs.add(arg)
            
            is_output = False
            for user in node.users:
                if user not in all_nodes:
                    is_output = True
                    break
            if is_output:
                outputs.add(node)
                
        partition.inputs = inputs
        partition.outputs = outputs
        self.candidates.append(partition)

    def _prune_candidates(self):
        min_nodes = _env_int("TILELANG_FRONTEND_MIN_NODES", 2)
        min_elems = _env_int("TILELANG_FRONTEND_MIN_ELEMS", 4096)
        # Default is intentionally low: many worthwhile subgraphs (e.g. softmax-like) have low AI.
        # Users can raise this to avoid memory-bound kernels.
        min_ai = _env_float("TILELANG_FRONTEND_MIN_AI", 0.5)
        require_single_output = os.environ.get("TILELANG_FRONTEND_REQUIRE_SINGLE_OUTPUT", "0").strip() == "1"
        validate_codegen = os.environ.get("TILELANG_FRONTEND_VALIDATE_CODEGEN", "0").strip() == "1"
        validate_topk = _env_int("TILELANG_FRONTEND_VALIDATE_CODEGEN_TOPK", 50)
        dump_prune = os.environ.get("TILELANG_FRONTEND_DUMP_PRUNE_STATS", "0").strip() == "1"

        # Reason counters (debug)
        reason = {
            "too_small": 0,
            "illegal_node": 0,
            "loop_incompat": 0,
            "bad_io": 0,
            "multi_output": 0,
            "too_few_elems": 0,
            "low_ai": 0,
            "kept": 0,
        }
        valid_candidates = []
        for p in self.candidates:
            # Allow single-node partitions for GEMM/bmm/linear (always worth replacing)
            has_gemm = any(
                ("mm" in (getattr(n.target, "__name__", "") or str(n.target)).lower()
                 or "linear" in (getattr(n.target, "__name__", "") or str(n.target)).lower()
                 or "einsum" in (getattr(n.target, "__name__", "") or str(n.target)).lower())
                for n in p.nodes if n.op == "call_function"
            )
            effective_min = 1 if has_gemm else min_nodes
            if len(p.nodes) < effective_min:
                reason["too_small"] += 1
                continue

            # Basic legality: only supported compute nodes, no boundary nodes.
            if any(self.support.is_any_boundary(n) or (not self.support.is_supported_inside_partition(n)) for n in p.nodes):
                reason["illegal_node"] += 1
                continue
            
            # Use LoopAnalyzer for check
            if not p.check_loop_compatibility():
                # print(f"Partition pruned due to loop conflict: {p.nodes}")
                reason["loop_incompat"] += 1
                continue

            # Avoid nasty boundaries at partition IO (ahlo-style pruning).
            # Hard boundary *outputs* that feed into external hard boundaries are problematic
            # (layout mismatch). But hard boundary *inputs* are fine — they just produce
            # tensors that the partition consumes.
            bad_io = False
            if not bad_io:
                for out in p.outputs:
                    for user in out.users:
                        if user in p.nodes:
                            continue
                        if self.support.is_hard_boundary(user):
                            bad_io = True
                            break
                    if bad_io:
                        break
            if bad_io:
                reason["bad_io"] += 1
                continue

            if require_single_output and len(p.outputs) != 1:
                reason["multi_output"] += 1
                continue

            # Skip remaining pruning for partitions containing GEMM/einsum (always worth fusing)
            if has_gemm:
                valid_candidates.append(p)
                reason["kept"] += 1
                continue

            # Scale/AI pruning: avoid tiny or memory-bound partitions that won't be worth a kernel.
            work = p.estimate_work_items()
            if work is not None and work < min_elems:
                reason["too_few_elems"] += 1
                continue
            ai = p.calculate_arithmetic_intensity()
            if ai < min_ai:
                reason["low_ai"] += 1
                continue

            # Optional: validate that our current codegen can actually lower this candidate.
            # This turns many "skip partition (codegen failed)" runtime surprises into early pruning.
            if validate_codegen:
                # Only validate larger candidates first to control overhead.
                # We'll validate later ones only if they are in the top-K by size.
                pass

            reason["kept"] += 1
            valid_candidates.append(p)
                
        if validate_codegen and valid_candidates:
            # Validate top-K by node count
            ranked = sorted(valid_candidates, key=lambda pp: len(pp.nodes), reverse=True)[:validate_topk]
            ok = []
            for pp in valid_candidates:
                if pp not in ranked:
                    ok.append(pp)
                    continue
                try:
                    sub_gm = extract_subgraph_gm(self.gm, pp.nodes, pp.inputs, pp.outputs)
                    codegen = ElementwiseCodegen(sub_gm)
                    _ = codegen.generate(tune=False)
                    ok.append(pp)
                except Exception:
                    continue
            self.candidates = ok
        else:
            self.candidates = valid_candidates

        if dump_prune:
            print(
                "[Partitioner] Prune config:",
                {
                    "min_nodes": min_nodes,
                    "min_elems": min_elems,
                    "min_ai": min_ai,
                    "require_single_output": require_single_output,
                    "validate_codegen": validate_codegen,
                    "validate_topk": validate_topk,
                },
            )
            print("[Partitioner] Prune stats:", reason)
            # Print a few kept candidates for sanity
            for i, p in enumerate(self.candidates[:5]):
                outs = [n.name for n in p.outputs]
                ins = [n.name for n in p.inputs]
                print(f"[Partitioner] Kept[{i}] nodes={len(p.nodes)} inputs={ins} outputs={outs} AI={p.calculate_arithmetic_intensity():.2f}")

    def _score_candidates_with_analyzer(self):
        """
        Estimate candidate cost using tilelang.tools.Analyzer and use it as the selection score.
        This is a prototype: we only analyze top-K (by node count) and only for candidates that can be
        lowered by our demo codegen.
        """
        topk = int(os.environ.get("TILELANG_FRONTEND_ANALYZER_TOPK", "20"))
        # Start from larger candidates (more likely to matter)
        ranked = sorted(self.candidates, key=lambda p: len(p.nodes), reverse=True)[:topk]

        try:
            device = CUDA("cuda")
        except Exception:
            # If no CUDA, fall back to heuristic
            for p in self.candidates:
                p.score = float(len(p.nodes))
            return

        for p in self.candidates:
            p.score = float(len(p.nodes))

        for p in ranked:
            # Prototype restriction: only score partitions whose inputs are true graph inputs
            # (placeholders). This avoids shape/layout mismatches when intermediate tensors
            # (e.g., keepdim reductions with shape [rows, 1]) become explicit inputs.
            if any(n.op != "placeholder" for n in p.inputs):
                continue

            # Build subgraph GM
            try:
                sub_gm = extract_subgraph_gm(self.gm, p.nodes, p.inputs, p.outputs)
            except Exception:
                continue

            # Only handle single-output + simple pointwise/reduction demo path
            out_rank = None
            for n in sub_gm.graph.nodes:
                if n.op == "output":
                    out = n.args[0]
                    if isinstance(out, tuple) and len(out) != 1:
                        out_rank = None
                    break

            # Infer a "shape signature" from first example input if possible
            rows = cols = total = None
            if self.example_inputs:
                inp0 = self.example_inputs[0]
                if hasattr(inp0, "shape"):
                    if inp0.dim() >= 2:
                        cols = int(inp0.shape[-1])
                        rows = int(inp0.numel() // cols)
                    else:
                        total = int(inp0.numel())

            # Lower with our demo codegen and get TIR
            try:
                codegen = ElementwiseCodegen(sub_gm)
                jit_impl = codegen.generate(tune=False)

                if codegen.has_reduction:
                    if rows is None or cols is None:
                        continue
                    key = (id(p), rows, cols, 1)
                    if key in self._analyzer_cache:
                        est_s = self._analyzer_cache[key]
                    else:
                        prim = jit_impl.get_tir(rows, cols)
                        res = Analyzer.analysis(prim, device)
                        est_s = float(res.estimated_time)
                        self._analyzer_cache[key] = est_s
                else:
                    if total is None:
                        # Flattened elementwise
                        if rows is not None and cols is not None:
                            total = rows * cols
                        else:
                            continue
                    key = (id(p), total, 0, 0)
                    if key in self._analyzer_cache:
                        est_s = self._analyzer_cache[key]
                    else:
                        prim = jit_impl.get_tir(total)
                        res = Analyzer.analysis(prim, device)
                        est_s = float(res.estimated_time)
                        self._analyzer_cache[key] = est_s

                # Score: prefer larger coverage (reduce launch+materialization overhead).
                # Use quadratic node weighting + penalize multiple outputs, and boost partitions
                # that directly produce the graph output.
                est_s = max(est_s, 1e-12)
                base = float(len(p.nodes) ** 2) / est_s
                base = base / max(1.0, float(len(p.outputs)))
                if any(o in self.graph_output_nodes for o in p.outputs):
                    base *= 2.0
                p.score = base
            except Exception:
                continue
