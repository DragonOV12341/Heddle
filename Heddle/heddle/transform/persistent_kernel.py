"""Persistent kernel transform for TileLang.

Transforms a standard kernel into a persistent kernel where CTAs remain
resident on SMs and process multiple tiles via dynamic scheduling.

Architecture:
    DSL -> TIR -> PCWS (normal) -> PersistentKernelTransform -> codegen

The transform:
1. Wraps the kernel body in a persistent tile-scheduling loop
2. Replaces blockIdx.{x,y,z} with locally computed tile coordinates
3. Adds atomicAdd-based dynamic tile distribution
4. Inserts mbarrier re-initialization between tiles

Usage:
    with PassContext(config={
        "tl.persistent_kernel": True,
        "tl.persistent_num_sms": 132,  # H100 SXM5
    }):
        mod = tilelang.transform.PersistentKernelTransform()(mod)

Or via DSL:
    with T.Kernel(M // bM, N // bN, threads=256, persistent=True):
        ...
"""

from __future__ import annotations

import os
from typing import Optional

from tilelang import tvm as tvm


def _get_int_config(ctx: tvm.ir.transform.PassContext, key: str, default: int) -> int:
    try:
        from heddle._monkey_patch import get_heddle_pass_config
        val = ctx.config.get(key, get_heddle_pass_config(key, default))
        return int(val)
    except Exception:
        return default


def _get_bool_config(ctx: tvm.ir.transform.PassContext, key: str, default: bool) -> bool:
    try:
        from heddle._monkey_patch import get_heddle_pass_config
        return bool(ctx.config.get(key, get_heddle_pass_config(key, default)))
    except Exception:
        return default


class PersistentKernelInfo:
    """Collected information for persistent kernel generation.

    Stored as an annotation on the kernel's top-level Block so that
    the CUDA codegen can emit the persistent tile-scheduling wrapper.
    """

    def __init__(
        self,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        num_sms: int,
        l2_swizzle: bool = False,
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.num_sms = num_sms
        self.l2_swizzle = l2_swizzle
        self.total_tiles = grid_x * grid_y * grid_z

    def to_annotation_dict(self) -> dict:
        return {
            "tl_persistent_grid_x": self.grid_x,
            "tl_persistent_grid_y": self.grid_y,
            "tl_persistent_grid_z": self.grid_z,
            "tl_persistent_num_sms": self.num_sms,
            "tl_persistent_total_tiles": self.total_tiles,
            "tl_persistent_l2_swizzle": int(self.l2_swizzle),
        }


def _detect_persistent_annotation(func: tvm.tir.PrimFunc) -> bool:
    """Check if the kernel has persistent=True annotation."""
    # Check function attrs
    if func.attrs and "persistent" in func.attrs:
        try:
            return bool(int(func.attrs["persistent"]))
        except (TypeError, ValueError):
            return False
    # Also check the top-level Block annotations
    body = func.body
    while isinstance(body, (tvm.tir.AttrStmt, tvm.tir.LetStmt)):
        body = body.body
    if isinstance(body, tvm.tir.BlockRealize):
        block = body.block
        if block.annotations and "persistent" in block.annotations:
            try:
                return bool(int(block.annotations["persistent"]))
            except (TypeError, ValueError):
                return False
    return False


def _extract_grid_dims(func: tvm.tir.PrimFunc):
    """Extract grid dimensions from the kernel's block iter_vars.

    Returns (grid_x, grid_y, grid_z) or None if not determinable.
    """
    body = func.body
    # Navigate through AttrStmt/LetStmt wrappers
    while isinstance(body, (tvm.tir.AttrStmt, tvm.tir.LetStmt)):
        body = body.body
    if not isinstance(body, tvm.tir.BlockRealize):
        return None

    block = body.block
    grid_dims = [1, 1, 1]  # x, y, z

    # Find blockIdx bindings in the kernel
    for iv in block.iter_vars:
        tag = iv.thread_tag if hasattr(iv, 'thread_tag') else ""
        if not tag:
            # Check AttrStmt bindings
            continue
        try:
            extent = int(iv.dom.extent)
        except (TypeError, ValueError):
            continue
        if "blockIdx.x" in tag:
            grid_dims[0] = extent
        elif "blockIdx.y" in tag:
            grid_dims[1] = extent
        elif "blockIdx.z" in tag:
            grid_dims[2] = extent

    # Also check function attrs for grid info
    if func.attrs:
        for attr_key in ["grid_x", "grid_y", "grid_z"]:
            if attr_key in func.attrs:
                try:
                    idx = ["grid_x", "grid_y", "grid_z"].index(attr_key)
                    grid_dims[idx] = int(func.attrs[attr_key])
                except (TypeError, ValueError):
                    pass

    return tuple(grid_dims)


def _get_default_num_sms() -> int:
    """Get the default number of SMs for the target GPU.

    Returns 132 for H100 SXM5 (most common Hopper target).
    """
    # Could query CUDA runtime, but for now use H100 default
    return 132


def PersistentKernelTransform():
    """Create a TVM pass that annotates kernels for persistent execution.

    This pass runs AFTER ProducerConsumerWarpSpecialized. It annotates
    the kernel with persistent scheduling metadata that the CUDA codegen
    uses to generate the persistent tile-scheduling wrapper.

    The actual code generation (atomicAdd tile counter, blockIdx replacement,
    mbarrier re-init) is handled by the codegen, not this pass. This pass
    only validates feasibility and attaches annotations.
    """

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="tl.PersistentKernelTransform")
    def _pass_func(func: tvm.tir.PrimFunc, mod, ctx: tvm.ir.transform.PassContext):
        enable = _get_bool_config(ctx, "tl.persistent_kernel", False)
        if not enable and not _detect_persistent_annotation(func):
            return func

        # Read num_sms: prefer DSL attr > pass config > default
        num_sms = 0
        if func.attrs and "persistent_num_sms" in func.attrs:
            try:
                num_sms = int(func.attrs["persistent_num_sms"])
            except (TypeError, ValueError):
                pass
        if num_sms <= 0:
            num_sms = _get_int_config(ctx, "tl.persistent_num_sms", 0)
        if num_sms <= 0:
            num_sms = _get_default_num_sms()

        # Read l2_swizzle: prefer DSL attr > pass config
        l2_swizzle = False
        if func.attrs and "persistent_l2_swizzle" in func.attrs:
            try:
                l2_swizzle = bool(int(func.attrs["persistent_l2_swizzle"]))
            except (TypeError, ValueError):
                pass
        if not l2_swizzle:
            l2_swizzle = _get_bool_config(ctx, "tl.persistent_l2_swizzle", False)

        # Extract grid dimensions
        grid = _extract_grid_dims(func)
        if grid is None:
            # Cannot determine grid dims — skip persistent transform
            return func

        grid_x, grid_y, grid_z = grid
        total_tiles = grid_x * grid_y * grid_z

        if total_tiles <= num_sms:
            # Not enough tiles to benefit from persistent scheduling
            return func

        info = PersistentKernelInfo(
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            num_sms=num_sms,
            l2_swizzle=l2_swizzle,
        )

        # Add persistent kernel annotations to the function
        new_attrs = dict(func.attrs.items()) if func.attrs else {}
        new_attrs.update(info.to_annotation_dict())
        new_attrs["tl_persistent"] = 1

        # Add tile_counter as a PrimFunc parameter so BOTH the tvm_ffi
        # and source-wrapper backends see it. The codegen emits the
        # persistent tile loop wrapper around the body.
        tile_counter_var = tvm.tir.Var("__tl_tile_counter", "handle")
        tile_counter_buf = tvm.tir.decl_buffer(
            [1], "int32", "__tl_tile_counter_buf",
            data=tile_counter_var, scope="global",
        )
        new_params = list(func.params) + [tile_counter_var]
        new_buffer_map = dict(func.buffer_map)
        new_buffer_map[tile_counter_var] = tile_counter_buf

        # Rewrite blockIdx thread_extent bindings to persistent grid.
        # This ensures lower_device_kernel_launch.cc picks up the
        # correct grid (num_sms, 1, 1) for the host launcher.
        persistent_grid_map = {
            "blockIdx.x": num_sms,
            "blockIdx.y": 1,
            "blockIdx.z": 1,
        }

        def _rewrite_grid(stmt):
            """Rewrite blockIdx AttrStmt extents to persistent grid."""
            if isinstance(stmt, tvm.tir.AttrStmt):
                if str(stmt.attr_key) == "thread_extent":
                    node = stmt.node
                    if hasattr(node, 'thread_tag'):
                        tag = str(node.thread_tag)
                        if tag in persistent_grid_map:
                            new_extent = persistent_grid_map[tag]
                            new_iv = tvm.tir.IterVar(
                                tvm.ir.Range(0, new_extent),
                                node.var, node.iter_type, node.thread_tag,
                            )
                            new_body = tvm.tir.stmt_functor.ir_transform(
                                stmt.body, None, _rewrite_grid,
                                ["tir.AttrStmt"],
                            )
                            return tvm.tir.AttrStmt(
                                new_iv, stmt.attr_key,
                                tvm.tir.IntImm("int32", new_extent),
                                new_body,
                            )
                return None  # unchanged, recurse normally
            return None

        new_body = tvm.tir.stmt_functor.ir_transform(
            func.body, None, _rewrite_grid, ["tir.AttrStmt"],
        )

        # Rebuild function with new parameter, body, and annotations
        new_func = tvm.tir.PrimFunc(
            new_params,
            new_body,
            func.ret_type,
            new_buffer_map,
            func.attrs,
        )
        return new_func.with_attrs(new_attrs)

    return _pass_func
