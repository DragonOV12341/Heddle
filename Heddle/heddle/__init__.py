"""Heddle: Automated async pipeline scheduling for TileLang.

Usage:
    import heddle
    heddle.init()  # patches TileLang at runtime

Then use TileLang as usual with Heddle pass configs:
    import tilelang
    from tilelang.transform import PassConfigKey

    pc = {
        PassConfigKey.TL_ENABLE_FAST_MATH: True,
        PassConfigKey.TL_ENABLE_AUTO_TL_PIPELINE_SMT: True,
        PassConfigKey.TL_ENABLE_HEDDLE_CONSUMER_SCHEDULE: True,
    }
    kern = tilelang.jit(out_idx=[...], pass_configs=pc)(my_kernel)
"""

__version__ = "0.1.0"


def init():
    """Apply Heddle monkey-patches to the active TileLang installation.

    Call once before using Heddle features. Safe to call multiple times.
    """
    from heddle._monkey_patch import apply_all_patches
    apply_all_patches()
