"""Runtime monkey-patches for integrating Heddle into stock TileLang 0.1.8.

Patches:
  1. phase.py — redirect HeddleConsumerSchedule import to heddle package
  2. pass_config.py — extend PassConfigKey enum with Heddle config keys
"""
from __future__ import annotations

import logging
import threading

_log = logging.getLogger("heddle")
_patched = False
_pass_context_patched = False
_pass_context_local = threading.local()
_pass_context_configs: dict[int, dict[str, object]] = {}

# ── Heddle pass-config keys ──────────────────────────────────────────────
# These must match the C++ registrations in builtin.cc.
HEDDLE_CONFIG_KEYS: dict[str, str] = {
    "TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE": "tl.enable_heddle_consumer_schedule",
    "TL_HEDDLE_USE_PRECISE_LATENCY": "tl.heddle_use_precise_latency",
    "TL_HEDDLE_USE_ALAP_PRIORITY": "tl.heddle_use_alap_priority",
    "TL_HEDDLE_BUFFER_SPAN_AWARE": "tl.heddle_buffer_span_aware",
    "TL_HEDDLE_USE_PHASE_B": "tl.heddle_use_phase_b",
    "TL_HEDDLE_PC_TOTAL_NUM_WARPS": "tl.heddle_pc_total_num_warps",
    "TL_HEDDLE_RELAX_PRODUCER_BOUNDARY": "tl.heddle_relax_producer_boundary",
    "TL_PCWS_ENABLE_THREE_ROLE": "tl.pcws_enable_three_role",
    "TL_PCWS_PRODUCER_THREAD_EXTENT": "tl.pcws_producer_thread_extent",
    "TL_FINEGRAINEDWS_ENABLE_THREE_ROLE": "tl.finegrainedws_enable_three_role",
    "TL_FINEGRAINEDWS_PRODUCER_THREAD_EXTENT": "tl.finegrainedws_producer_thread_extent",
    "TL_ENABLE_AUTO_TL_PIPELINE_SMT": "tl.enable_auto_tl_pipeline_smt",
    "TL_OVERWRITE_AUTO_TL_PIPELINE_ANNOTATIONS": "tl.overwrite_auto_tl_pipeline_annotations",
    "TL_AUTO_TL_PIPELINE_NUM_STAGES": "tl.auto_tl_pipeline_num_stages",
    "TL_AUTO_TL_PIPELINE_SMT_MULTISTAGE": "tl.auto_tl_pipeline_smt_multistage",
    "TL_SMT_USE_PRECISE_LATENCY": "tl.smt_use_precise_latency",
    "TL_SMT_FORCE_GROUP": "tl.smt_force_group",
    "TL_SMT_FORCE_ORDER": "tl.smt_force_order",
    "TL_SMT_FORCE_STAGE": "tl.smt_force_stage",
    "TL_SMT_SEARCH_ORDER": "tl.smt_search_order",
    "TL_SMT_STAGE_OFFSET": "tl.smt_stage_offset",
    "TL_SMT_DISALLOW_SPILLS": "tl.smt_disallow_spills",
    "TL_SMT_USE_SPILL_CONCURRENCY": "tl.smt_use_spill_concurrency",
    "TL_SMT_INCLUDE_INCOMING_LIVE": "tl.smt_include_incoming_live",
    "TL_SMT_PATTERN_SEARCH_ENABLE": "tl.smt_pattern_search_enable",
    "TL_PCWS_DUAL_CONSUMER": "tl.pcws_dual_consumer",
    "TL_PCWS_CONSUMER_STAGE_MAP": "tl.pcws_consumer_stage_map",
    "TL_PCWS_BARRIER_HINTS": "tl.pcws_barrier_hints",
    "TL_PCWS_STAGE_OFFSETS": "tl.pcws_stage_offsets",
    "TL_FINEGRAINEDWS_DUAL_CONSUMER": "tl.finegrainedws_dual_consumer",
    "TL_FINEGRAINEDWS_CONSUMER_STAGE_MAP": "tl.finegrainedws_consumer_stage_map",
    "TL_FINEGRAINEDWS_BARRIER_HINTS": "tl.finegrainedws_barrier_hints",
    "TL_FINEGRAINEDWS_STAGE_OFFSETS": "tl.finegrainedws_stage_offsets",
    "TL_PCWS_WARP_ASSIGNS": "tl.pcws_warp_assigns",
    "TL_FINEGRAINEDWS_WARP_ASSIGNS": "tl.finegrainedws_warp_assigns",
    "TL_PERSISTENT_KERNEL": "tl.persistent_kernel",
    "TL_PERSISTENT_NUM_SMS": "tl.persistent_num_sms",
    "TL_PERSISTENT_L2_SWIZZLE": "tl.persistent_l2_swizzle",
}

HEDDLE_CONFIG_VALUES = frozenset(HEDDLE_CONFIG_KEYS.values())


def _add_enum_member(enum_cls, name: str, value: str) -> None:
    """Add a str Enum member after class creation."""
    member = str.__new__(enum_cls, value)
    object.__setattr__(member, "_name_", name)
    object.__setattr__(member, "_value_", value)
    enum_cls._member_names_.append(name)
    enum_cls._member_map_[name] = member
    enum_cls._value2member_map_[value] = member
    type.__setattr__(enum_cls, name, member)


def _patch_pass_config() -> None:
    """Extend PassConfigKey enum with Heddle-specific keys."""
    try:
        from tilelang.transform.pass_config import PassConfigKey
    except ImportError:
        _log.warning("tilelang.transform.pass_config not found; skipping enum extension")
        return

    for name, value in HEDDLE_CONFIG_KEYS.items():
        if hasattr(PassConfigKey, name):
            continue
        _add_enum_member(PassConfigKey, name, value)


def _config_key_to_string(key) -> str:
    if hasattr(key, "value"):
        return str(key.value)
    return str(key)


def get_heddle_pass_config(key: str, default=None):
    """Return a Heddle pass config stripped from the active TVM PassContext."""
    stack = getattr(_pass_context_local, "stack", None)
    if not stack:
        return default
    for config in reversed(stack):
        if key in config:
            return config[key]
    return default


def _patch_pass_context() -> None:
    """Let stock TileLang accept Heddle pass configs without C++ registration."""
    global _pass_context_patched
    if _pass_context_patched:
        return

    try:
        from tilelang import tvm
    except ImportError:
        _log.warning("tilelang.tvm not found; skipping PassContext patch")
        return

    pass_context_cls = tvm.transform.PassContext
    original_init = pass_context_cls.__init__
    original_enter = pass_context_cls.__enter__
    original_exit = pass_context_cls.__exit__

    def _patched_init(self, *args, **kwargs):
        config = kwargs.get("config", None)
        if len(args) >= 5:
            config = args[4]

        heddle_config: dict[str, object] = {}
        if config:
            clean_config = {}
            for key, value in dict(config).items():
                key_str = _config_key_to_string(key)
                if key_str in HEDDLE_CONFIG_VALUES:
                    heddle_config[key_str] = value
                else:
                    clean_config[key] = value

            if len(args) >= 5:
                args = list(args)
                args[4] = clean_config or None
                args = tuple(args)
            else:
                kwargs["config"] = clean_config or None

        original_init(self, *args, **kwargs)
        _pass_context_configs[id(self)] = heddle_config

    def _patched_enter(self):
        stack = getattr(_pass_context_local, "stack", None)
        if stack is None:
            stack = []
            _pass_context_local.stack = stack
        stack.append(_pass_context_configs.get(id(self), {}))
        try:
            return original_enter(self)
        except Exception:
            stack.pop()
            raise

    def _patched_exit(self, ptype, value, trace):
        try:
            return original_exit(self, ptype, value, trace)
        finally:
            stack = getattr(_pass_context_local, "stack", None)
            if stack:
                stack.pop()
            _pass_context_configs.pop(id(self), None)

    pass_context_cls.__init__ = _patched_init
    pass_context_cls.__enter__ = _patched_enter
    pass_context_cls.__exit__ = _patched_exit
    _pass_context_patched = True
    _log.info("Patched TVM PassContext for Heddle pass configs")


def _patch_phase() -> None:
    """Keep TileLang's native lowering pipeline intact.

    Heddle is inserted by wrapping ProducerConsumerWarpSpecialized in
    _patch_transform_init().  Replacing OptimizeForTarget is too brittle
    because TileLang's native pipeline contains many post-WS cleanups.
    """
    try:
        import tilelang.engine.phase as phase_mod
    except ImportError:
        _log.warning("tilelang.engine.phase not found; skipping phase patch")
        return

    _log.info("Using TileLang native OptimizeForTarget; Heddle wraps PCWS pass")


def _patch_transform_init() -> None:
    """Add Heddle pass constructors to tilelang.transform namespace."""
    try:
        import tilelang.transform as transform_mod
    except ImportError:
        return

    from heddle.transform.heddle_consumer_schedule import HeddleConsumerSchedule
    from tilelang import tvm as tvm

    if not hasattr(transform_mod, 'HeddleConsumerSchedule'):
        transform_mod.HeddleConsumerSchedule = HeddleConsumerSchedule
    
    # 将 SMT 分析放到MVB前面进行。排除 用户的 numstage对 IR结构的影响
    if not hasattr(transform_mod, "_heddle_original_pcws"):
        transform_mod._old_mvb = transform_mod.MultiVersionBuffer
        def _mvb_wrapper():
            return tvm.transform.Sequential([
                HeddleConsumerSchedule(),
                transform_mod._old_mvb(),
            ])

        transform_mod.MultiVersionBuffer = _mvb_wrapper
    # if not hasattr(transform_mod, "_heddle_original_pcws"):
    #     transform_mod._heddle_original_pcws = transform_mod.ProducerConsumerWarpSpecialized

    #     def _heddle_pcws_wrapper():
    #         return tvm.transform.Sequential([
    #             HeddleConsumerSchedule(),
    #             transform_mod._heddle_original_pcws(),
    #         ])

    #     transform_mod.ProducerConsumerWarpSpecialized = _heddle_pcws_wrapper


def apply_all_patches() -> None:
    global _patched
    if _patched:
        return
    _patch_pass_config()
    _patch_pass_context()
    _patch_phase()
    _patch_transform_init()
    _patched = True
    _log.info("Heddle patches applied")
