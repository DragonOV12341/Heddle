import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import numpy as np

# ----------------- 1. 原始数据与配置 -----------------

def configure_chinese_font():
    """Prefer a CJK-capable font so Chinese text in op_desc can render."""
    font_path_candidates = []
    env_font_path = os.environ.get('CJK_FONT_PATH')
    if env_font_path:
        font_path_candidates.append(env_font_path)
    font_path_candidates.extend([
        os.path.join(os.path.dirname(__file__), 'fonts', filename)
        for filename in (
            'NotoSansCJK-Regular.ttc',
            'NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf',
            'SourceHanSansCN-Regular.otf',
            'WenQuanYiMicroHei.ttf',
            'SimHei.ttf',
        )
    ])

    for font_path in font_path_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            selected_font = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Using CJK font: {selected_font} ({font_path})")
            return

    font_candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Source Han Sans SC',
        'Source Han Sans CN',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Microsoft YaHei',
        'SimHei',
        'Arial Unicode MS',
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in font_candidates if font in installed_fonts), None)
    if selected_font is None:
        print(
            "Warning: no CJK font found. Chinese text in op_desc may render as boxes. "
            "Install a font such as Noto Sans CJK SC, set CJK_FONT_PATH, "
            "or put a supported font file under ./fonts/."
        )
        return

    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


configure_chinese_font()

# optimized mod sched
# warp_assign={'s0': 6, 's2': 2, 's3': 2, 's4': 2, 's5': 0, 's6': 3, 's7': 0, 's8': 3, 's9': 1, 's10': 2, 's11': 3, 's12': 2, 's13': 0, 's14': 6, 's16': 0}
# optimized_L=1174
# base_I=512
# optimized_M={0: 0, 2: 8, 3: 4, 4: 12, 5: 282, 6: 434, 7: 440, 8: 448, 9: 448, 10: 978, 11: 498, 12: 984, 13: 978, 14: 2, 16: 1046}

# warp_assign={'s0': 1, 's2': 11, 's3': 8, 's4': 7, 's5': 8, 's6': 11, 's7': 11, 's8': 5, 's9': 10, 's10': 9, 's11': 9, 's12': 9, 's13': 11, 's14': 0, 's16': 4}
# optimized_L=754
# base_I=257
# optimized_M={0: 0, 2: 0, 3: 6, 4: 2, 5: 282, 6: 370, 7: 374, 8: 380, 9: 380, 10: 654, 11: 414, 12: 658, 13: 654, 14: 0, 16: 690}

warp_assign={'s0__u0': 7, 's0__u1': 7, 's2': 3, 's3': 3, 's4__u0': 3, 's4__u1': 3, 's4__u2': 3, 's4__u3': 3, 's4__u4': 3, 's4__u5': 3, 's4__u6': 3, 's4__u7': 3, 's5__u0': 0, 's5__u1': 0, 's5__u2': 0, 's5__u3': 0, 's5__u4': 0, 's5__u5': 0, 's5__u6': 0, 's5__u7': 0, 's6__u0': 1, 's6__u1': 1, 's7__u0': 0, 's7__u1': 1, 's8__u0': 3, 's8__u1': 1, 's9__u0': 1, 's9__u1': 1, 's9__u2': 1, 's9__u3': 1, 's9__u4': 1, 's9__u5': 1, 's9__u6': 1, 's9__u7': 1, 's9__u8': 1, 's9__u9': 1, 's9__u10': 1, 's9__u11': 1, 's9__u12': 1, 's9__u13': 1, 's9__u14': 1, 's9__u15': 1, 's9__u16': 1, 's9__u17': 1, 's9__u18': 1, 's9__u19': 1, 's9__u20': 1, 's9__u21': 1, 's9__u22': 1, 's9__u23': 1, 's9__u24': 1, 's9__u25': 1, 's9__u26': 1, 's9__u27': 1, 's9__u28': 1, 's9__u29': 1, 's9__u30': 1, 's9__u31': 1, 's10__u0': 3, 's10__u1': 1, 's11__u0': 1, 's11__u1': 1, 's11__u2': 1, 's11__u3': 1, 's11__u4': 1, 's11__u5': 1, 's11__u6': 1, 's11__u7': 1, 's11__u8': 1, 's11__u9': 1, 's11__u10': 1, 's11__u11': 1, 's11__u12': 1, 's11__u13': 1, 's11__u14': 1, 's11__u15': 1, 's11__u16': 1, 's11__u17': 1, 's11__u18': 1, 's11__u19': 1, 's11__u20': 1, 's11__u21': 1, 's11__u22': 1, 's11__u23': 1, 's11__u24': 1, 's11__u25': 1, 's11__u26': 1, 's11__u27': 1, 's11__u28': 1, 's11__u29': 1, 's11__u30': 1, 's11__u31': 1, 's11__u32': 1, 's11__u33': 1, 's11__u34': 1, 's11__u35': 1, 's11__u36': 1, 's11__u37': 1, 's11__u38': 1, 's11__u39': 1, 's11__u40': 1, 's11__u41': 1, 's11__u42': 1, 's11__u43': 1, 's11__u44': 1, 's11__u45': 1, 's11__u46': 1, 's11__u47': 1, 's11__u48': 1, 's11__u49': 1, 's11__u50': 1, 's11__u51': 1, 's11__u52': 1, 's11__u53': 1, 's11__u54': 1, 's11__u55': 1, 's11__u56': 1, 's11__u57': 1, 's11__u58': 1, 's11__u59': 1, 's11__u60': 1, 's11__u61': 1, 's11__u62': 1, 's11__u63': 1, 's12__u0': 3, 's12__u1': 1, 's13__u0': 3, 's13__u1': 3, 's13__u2': 3, 's13__u3': 3, 's13__u4': 3, 's13__u5': 3, 's13__u6': 3, 's13__u7': 1, 's14__u0': 7, 's14__u1': 7, 's16__u0': 0, 's16__u1': 0, 's16__u2': 0, 's16__u3': 0}
optimized_L=802
base_I=232
optimized_M={0: 0, 2: 0, 3: 5, 4: 0, 5: 282, 6: 314, 7: 317, 8: 322, 9: 546, 10: 572, 11: 572, 12: 575, 13: 572, 14: 0, 16: 586}
expanded_M={'s0__u0': 0, 's0__u1': 1, 's2': 0, 's3': 5, 's4__u0': 0, 's4__u1': 0, 's4__u2': 0, 's4__u3': 0, 's4__u4': 0, 's4__u5': 0, 's4__u6': 0, 's4__u7': 0, 's5__u0': 282, 's5__u1': 314, 's5__u2': 346, 's5__u3': 378, 's5__u4': 410, 's5__u5': 442, 's5__u6': 474, 's5__u7': 506, 's6__u0': 314, 's6__u1': 538, 's7__u0': 317, 's7__u1': 541, 's8__u0': 322, 's8__u1': 546, 's9__u0': 546, 's9__u1': 546, 's9__u2': 546, 's9__u3': 546, 's9__u4': 546, 's9__u5': 554, 's9__u6': 546, 's9__u7': 546, 's9__u8': 554, 's9__u9': 546, 's9__u10': 546, 's9__u11': 546, 's9__u12': 562, 's9__u13': 562, 's9__u14': 546, 's9__u15': 546, 's9__u16': 546, 's9__u17': 546, 's9__u18': 554, 's9__u19': 554, 's9__u20': 546, 's9__u21': 554, 's9__u22': 554, 's9__u23': 554, 's9__u24': 554, 's9__u25': 554, 's9__u26': 554, 's9__u27': 554, 's9__u28': 554, 's9__u29': 554, 's9__u30': 554, 's9__u31': 554, 's10__u0': 572, 's10__u1': 588, 's11__u0': 572, 's11__u1': 572, 's11__u2': 572, 's11__u3': 572, 's11__u4': 573, 's11__u5': 572, 's11__u6': 572, 's11__u7': 572, 's11__u8': 572, 's11__u9': 572, 's11__u10': 572, 's11__u11': 572, 's11__u12': 572, 's11__u13': 572, 's11__u14': 572, 's11__u15': 572, 's11__u16': 572, 's11__u17': 572, 's11__u18': 572, 's11__u19': 572, 's11__u20': 572, 's11__u21': 572, 's11__u22': 572, 's11__u23': 572, 's11__u24': 572, 's11__u25': 572, 's11__u26': 572, 's11__u27': 573, 's11__u28': 572, 's11__u29': 572, 's11__u30': 572, 's11__u31': 572, 's11__u32': 573, 's11__u33': 572, 's11__u34': 572, 's11__u35': 572, 's11__u36': 572, 's11__u37': 572, 's11__u38': 572, 's11__u39': 572, 's11__u40': 572, 's11__u41': 572, 's11__u42': 572, 's11__u43': 572, 's11__u44': 572, 's11__u45': 572, 's11__u46': 572, 's11__u47': 572, 's11__u48': 572, 's11__u49': 572, 's11__u50': 572, 's11__u51': 572, 's11__u52': 572, 's11__u53': 572, 's11__u54': 572, 's11__u55': 572, 's11__u56': 572, 's11__u57': 572, 's11__u58': 573, 's11__u59': 573, 's11__u60': 572, 's11__u61': 572, 's11__u62': 572, 's11__u63': 572, 's12__u0': 575, 's12__u1': 591, 's13__u0': 572, 's13__u1': 572, 's13__u2': 573, 's13__u3': 572, 's13__u4': 572, 's13__u5': 580, 's13__u6': 573, 's13__u7': 588, 's14__u0': 0, 's14__u1': 1, 's16__u0': 586, 's16__u1': 650, 's16__u2': 714, 's16__u3': 786}

# naive mod sched
I = 257
L = 754
M = {0: 0, 2: 358, 3: 364, 4: 246, 5: 282, 6: 370, 7: 374, 8: 588, 9: 380, 10: 748, 11: 622, 12: 752, 13: 654, 14: 408, 16: 690}

# ========================================
# 执行+发射延迟
# latencies={0: 282, 1: 1, 2: 8, 3: 8, 4: 68, 5: 152, 6: 6, 7: 8, 8: 50, 9: 530, 10: 6, 11: 132, 12: 8, 13: 68, 14: 282, 15: 1, 16: 176}
# 发射延迟
# duration={0: 2, 1: 1, 2: 4, 3: 4, 4: 64, 5: 128, 6: 4, 7: 4, 8: 32, 9: 512, 10: 4, 11: 128, 12: 4, 13: 64, 14: 2, 15: 1, 16: 128}

latencies={(0, 0): 281, (0, 1): 281, (2, 0): 5, (3, 0): 5, (4, 0): 5, (4, 1): 5, (4, 2): 5, (4, 3): 5, (4, 4): 5, (4, 5): 5, (4, 6): 5, (4, 7): 5, (5, 0): 32, (5, 1): 32, (5, 2): 32, (5, 3): 32, (5, 4): 32, (5, 5): 32, (5, 6): 32, (5, 7): 32, (6, 0): 3, (6, 1): 3, (7, 0): 5, (7, 1): 5, (8, 0): 26, (8, 1): 26, (9, 0): 26, (9, 1): 26, (9, 2): 26, (9, 3): 26, (9, 4): 26, (9, 5): 26, (9, 6): 26, (9, 7): 26, (9, 8): 26, (9, 9): 26, (9, 10): 26, (9, 11): 26, (9, 12): 26, (9, 13): 26, (9, 14): 26, (9, 15): 26, (9, 16): 26, (9, 17): 26, (9, 18): 26, (9, 19): 26, (9, 20): 26, (9, 21): 26, (9, 22): 26, (9, 23): 26, (9, 24): 26, (9, 25): 26, (9, 26): 26, (9, 27): 26, (9, 28): 26, (9, 29): 26, (9, 30): 26, (9, 31): 26, (10, 0): 3, (10, 1): 3, (11, 0): 5, (11, 1): 5, (11, 2): 5, (11, 3): 5, (11, 4): 5, (11, 5): 5, (11, 6): 5, (11, 7): 5, (11, 8): 5, (11, 9): 5, (11, 10): 5, (11, 11): 5, (11, 12): 5, (11, 13): 5, (11, 14): 5, (11, 15): 5, (11, 16): 5, (11, 17): 5, (11, 18): 5, (11, 19): 5, (11, 20): 5, (11, 21): 5, (11, 22): 5, (11, 23): 5, (11, 24): 5, (11, 25): 5, (11, 26): 5, (11, 27): 5, (11, 28): 5, (11, 29): 5, (11, 30): 5, (11, 31): 5, (11, 32): 5, (11, 33): 5, (11, 34): 5, (11, 35): 5, (11, 36): 5, (11, 37): 5, (11, 38): 5, (11, 39): 5, (11, 40): 5, (11, 41): 5, (11, 42): 5, (11, 43): 5, (11, 44): 5, (11, 45): 5, (11, 46): 5, (11, 47): 5, (11, 48): 5, (11, 49): 5, (11, 50): 5, (11, 51): 5, (11, 52): 5, (11, 53): 5, (11, 54): 5, (11, 55): 5, (11, 56): 5, (11, 57): 5, (11, 58): 5, (11, 59): 5, (11, 60): 5, (11, 61): 5, (11, 62): 5, (11, 63): 5, (12, 0): 5, (12, 1): 5, (13, 0): 5, (13, 1): 5, (13, 2): 5, (13, 3): 5, (13, 4): 5, (13, 5): 5, (13, 6): 5, (13, 7): 5, (14, 0): 281, (14, 1): 281, (16, 0): 64, (16, 1): 64, (16, 2): 64, (16, 3): 64}
duration={(0, 0): 1, (0, 1): 1, (2, 0): 1, (3, 0): 1, (4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 1, (4, 5): 1, (4, 6): 1, (4, 7): 1, (5, 0): 8, (5, 1): 8, (5, 2): 8, (5, 3): 8, (5, 4): 8, (5, 5): 8, (5, 6): 8, (5, 7): 8, (6, 0): 1, (6, 1): 1, (7, 0): 1, (7, 1): 1, (8, 0): 8, (8, 1): 8, (9, 0): 8, (9, 1): 8, (9, 2): 8, (9, 3): 8, (9, 4): 8, (9, 5): 8, (9, 6): 8, (9, 7): 8, (9, 8): 8, (9, 9): 8, (9, 10): 8, (9, 11): 8, (9, 12): 8, (9, 13): 8, (9, 14): 8, (9, 15): 8, (9, 16): 8, (9, 17): 8, (9, 18): 8, (9, 19): 8, (9, 20): 8, (9, 21): 8, (9, 22): 8, (9, 23): 8, (9, 24): 8, (9, 25): 8, (9, 26): 8, (9, 27): 8, (9, 28): 8, (9, 29): 8, (9, 30): 8, (9, 31): 8, (10, 0): 1, (10, 1): 1, (11, 0): 1, (11, 1): 1, (11, 2): 1, (11, 3): 1, (11, 4): 1, (11, 5): 1, (11, 6): 1, (11, 7): 1, (11, 8): 1, (11, 9): 1, (11, 10): 1, (11, 11): 1, (11, 12): 1, (11, 13): 1, (11, 14): 1, (11, 15): 1, (11, 16): 1, (11, 17): 1, (11, 18): 1, (11, 19): 1, (11, 20): 1, (11, 21): 1, (11, 22): 1, (11, 23): 1, (11, 24): 1, (11, 25): 1, (11, 26): 1, (11, 27): 1, (11, 28): 1, (11, 29): 1, (11, 30): 1, (11, 31): 1, (11, 32): 1, (11, 33): 1, (11, 34): 1, (11, 35): 1, (11, 36): 1, (11, 37): 1, (11, 38): 1, (11, 39): 1, (11, 40): 1, (11, 41): 1, (11, 42): 1, (11, 43): 1, (11, 44): 1, (11, 45): 1, (11, 46): 1, (11, 47): 1, (11, 48): 1, (11, 49): 1, (11, 50): 1, (11, 51): 1, (11, 52): 1, (11, 53): 1, (11, 54): 1, (11, 55): 1, (11, 56): 1, (11, 57): 1, (11, 58): 1, (11, 59): 1, (11, 60): 1, (11, 61): 1, (11, 62): 1, (11, 63): 1, (12, 0): 1, (12, 1): 1, (13, 0): 1, (13, 1): 1, (13, 2): 1, (13, 3): 1, (13, 4): 1, (13, 5): 1, (13, 6): 1, (13, 7): 1, (14, 0): 1, (14, 1): 1, (16, 0): 16, (16, 1): 16, (16, 2): 16, (16, 3): 16}


op_desc = {
    0 : "tma_copy Ks" ,
    1 : "wait Ks" ,
    2 : "smp = sm" ,
    3 : "clear sm" ,
    4 : "clear acc_s" ,
    5 : "acc_s = wgmma QK" ,
    6 : "softmax: reduce_max(QK)" ,
    7 : "softmax: update global max" ,
    8 : "softmax: get scale" ,
    9 : "softmax: exp" ,
    10 : "softmax: sumexp" ,
    11 : "rescale last PV" ,
    12 : "accumulate sumexp" ,
    13 : "f32tof16(P)" ,
    14 : "tma_copy Vs" ,
    15 : "wait Vs" ,
    16 : "acc_o += wgmma PV" ,
    # acc_o = acc_o / ls
    # copy acc_o to Output
}

unit_mapping = {
    'TMA': [0, 1, 14, 15],
    'TC': [5, 16, ],
    'ALU': [2, 3, 4, 6, 7, 10, 11, 12, 13, ],
    'SFU': [8, 9, ]
}

BOX_HEIGHT = 0.55
LIFETIME_BOX_HEIGHT = 0.28
LIFETIME_ROW_GAP = 1.15
LIFETIME_ITER_Y_STEP = 0.33
ISSUE_ALPHA = 0.6
UNITS_ORDER = ['TMA', 'TC', 'ALU', 'SFU']

ITERATION_COLOR_PALETTE = {
    1: {'face': '#5c3d2e', 'edge': '#ba7a5f'},  # i-1
    0: {'face': '#1f4e5b', 'edge': '#3a889e'},  # i
    2: {'face': '#3f4f2f', 'edge': '#8aaa5e'},  # i+1
}
STAGE_LABELS = {
    1: 'i-1',
    0: 'i',
    2: 'i+1',
}
WG_HATCHES = {
    0: '///',
    1: '',
    2: '...',
    3: '\\\\\\',
}
LIFETIME_ITER_COLORS = {
    -1: {'face': '#5c3d2e', 'edge': '#ba7a5f'},
    0: {'face': '#1f4e5b', 'edge': '#3a889e'},
    1: {'face': '#3f4f2f', 'edge': '#8aaa5e'},
}
LIFETIME_STORAGE_HATCHES = {
    'RMEM': '',
    'SMEM': '///',
}


def parse_op_key(op_key):
    """Return (op_id, unroll_id) for keys like 9, (9, 3), s9, or s9__u3."""
    if isinstance(op_key, tuple):
        return op_key
    if isinstance(op_key, int):
        return (op_key, None)
    if isinstance(op_key, str) and op_key.startswith('s'):
        name = op_key[1:]
        if '__u' in name:
            op_id, unroll_id = name.split('__u', 1)
            return (int(op_id), int(unroll_id))
        return (int(name), None)
    raise ValueError(f"Unsupported op key: {op_key!r}")


def format_op_key(op_key):
    op, unroll = parse_op_key(op_key)
    return f"s{op}" if unroll is None else f"s{op}__u{unroll}"


def op_key_sort_key(op_key):
    op, unroll = parse_op_key(op_key)
    return (op, -1 if unroll is None else unroll)


def build_fine_grained_schedule(schedule_m, fine_latency_table, fine_duration_table, *, allow_parent_fallback=False):
    """Expand op-level M into fine-grained op instances when needed."""
    fine_keys = sorted(
        set(fine_latency_table) | set(fine_duration_table),
        key=op_key_sort_key,
    )
    fine_key_count_by_op = {}
    for fine_key in fine_keys:
        op, _ = parse_op_key(fine_key)
        fine_key_count_by_op[op] = fine_key_count_by_op.get(op, 0) + 1

    fine_schedule = {}
    fallback_keys = []
    for fine_key in fine_keys:
        op, _ = parse_op_key(fine_key)
        exact_candidates = (fine_key, format_op_key(fine_key))
        parent_candidates = ((op, None), op, f"s{op}")
        can_use_parent = allow_parent_fallback or fine_key_count_by_op.get(op, 0) == 1
        candidates = exact_candidates + (parent_candidates if can_use_parent else ())
        for candidate in candidates:
            if candidate in schedule_m:
                fine_schedule[fine_key] = schedule_m[candidate]
                if (
                    candidate in parent_candidates
                    and parse_op_key(fine_key)[1] is not None
                    and fine_key_count_by_op.get(op, 0) > 1
                ):
                    fallback_keys.append(format_op_key(fine_key))
                break
        else:
            raise KeyError(f"Missing schedule time for {format_op_key(fine_key)} / parent s{op}")
    if fallback_keys:
        print(
            "Warning: using collapsed parent-level M for fine-grained rows; "
            "copy expanded_M from the latest log into expanded_M for dependency-accurate timing. "
            f"Fallback rows: {fallback_keys[:8]}{'...' if len(fallback_keys) > 8 else ''}"
        )
    return fine_schedule


def build_warpgroup_info(warp_assignment):
    """Return fine-op->warpgroup and warpgroup->op descriptions."""
    op_wgid = {parse_op_key(op_name): op_warp // 4 for op_name, op_warp in warp_assignment.items()}
    active_wgids = sorted(set(op_wgid.values()))
    wgid_ops = {wgid: [] for wgid in active_wgids}
    for op_key, wgid in sorted(op_wgid.items(), key=lambda item: op_key_sort_key(item[0])):
        op, _ = parse_op_key(op_key)
        wgid_ops[wgid].append(format_op_key(op_key))
    return op_wgid, active_wgids, wgid_ops


def get_warpgroup(op_wgid, op_key):
    op, _ = parse_op_key(op_key)
    return op_wgid.get(op_key, op_wgid.get((op, None)))


def build_y_layout(op_keys):
    """Build one stable y row for each fine-grained op, grouped by execution unit."""
    y_ticks_positions = []
    y_ticks_labels = []
    op_y_pos = {}
    current_y = 0
    unit_boundaries = []
    op_keys_by_unit = {unit: [] for unit in UNITS_ORDER}
    for op_key in sorted(op_keys, key=op_key_sort_key):
        op, _ = parse_op_key(op_key)
        for unit, unit_ops in unit_mapping.items():
            if op in unit_ops:
                op_keys_by_unit.setdefault(unit, []).append(op_key)
                break

    for unit in reversed(UNITS_ORDER):
        unit_ops = op_keys_by_unit.get(unit, [])
        if not unit_ops:
            continue

        start_y_for_unit = current_y
        for op_key in unit_ops:
            op, _ = parse_op_key(op_key)
            op_y_pos[op_key] = current_y
            y_ticks_positions.append(current_y)
            y_ticks_labels.append(f"{unit}: {format_op_key(op_key)}")
            current_y += 1

        end_y_for_unit = current_y - 1
        unit_boundaries.append((unit, start_y_for_unit, end_y_for_unit))
        current_y += 0.6

    return op_y_pos, y_ticks_positions, y_ticks_labels, unit_boundaries, current_y


def make_iteration_configs(ii):
    return [
        {'start': 0, 'end': ii, 'stage': 1},
        {'start': ii, 'end': ii * 2, 'stage': 0},
        {'start': ii * 2, 'end': ii * 3, 'stage': 2},
    ]


def build_overlap_items(schedule_m, ii, schedule_l, iteration_configs):
    plot_items = []
    all_events = {0, ii, ii * 2, ii * 3, ii + schedule_l, ii * 2 + schedule_l}

    # 这里画的是 logical iteration 的绝对 M 时间线，不能用 M % II。
    # M % II 只适合画资源槽位表；用于依赖时间线会把跨 II 的操作折回前面。
    for iteration in iteration_configs:
        for op_key, start_time in schedule_m.items():
            t_start = start_time + iteration['start']
            t_end = t_start + latencies[op_key]
            t_issue_end = t_start + duration[op_key]
            plot_items.append((op_key, t_start, t_end, t_issue_end, iteration['stage']))
            all_events.update((t_start, t_end, t_issue_end))

    sorted_events = sorted(all_events)
    time_to_coord = {t: i for i, t in enumerate(sorted_events)}
    return plot_items, sorted_events, time_to_coord


def setup_dark_axis(fig, ax):
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')


def draw_unit_guides(ax, event_count, unit_boundaries, y_ticks_positions, y_ticks_labels):
    for idx in range(event_count):
        ax.axvline(x=idx, color='#2c2c2c', linestyle=':', linewidth=1.0, zorder=1)

    for i in range(len(unit_boundaries) - 1):
        end_y = unit_boundaries[i][2]
        next_start_y = unit_boundaries[i + 1][1]
        sep_y = (end_y + next_start_y) / 2
        ax.axhline(y=sep_y, color='#555555', linestyle='--', linewidth=1.2, zorder=1)

    ax.set_yticks(y_ticks_positions)
    ax.set_yticklabels(y_ticks_labels, fontsize=9)


def choose_xticks(sorted_events, important_times, max_xtick_labels=34, min_tick_gap=3):
    tick_step = max(1, int(np.ceil(len(sorted_events) / max_xtick_labels)))
    important_positions = {
        idx
        for idx, t in enumerate(sorted_events)
        if t in important_times
    }
    sampled_positions = {
        idx
        for idx in range(len(sorted_events))
        if idx % tick_step == 0
    }
    xtick_positions = sorted(important_positions)
    for idx in sorted(sampled_positions):
        if all(abs(idx - important_idx) >= min_tick_gap for important_idx in important_positions):
            xtick_positions.append(idx)
    return sorted(set(xtick_positions))


def draw_schedule_block(ax, cx_start, cx_end, cx_issue_end, y_pos, colors, label, hatch=None):
    c_width = max(cx_end - cx_start, 0.25)
    issue_width = max(cx_issue_end - cx_start, 0.25)
    rect = patches.Rectangle(
        (cx_start, y_pos - BOX_HEIGHT / 2),
        c_width, BOX_HEIGHT,
        linewidth=1.5, edgecolor=colors['edge'], facecolor=colors['edge'], zorder=3, alpha=0.28
    )
    ax.add_patch(rect)

    issue_rect = patches.Rectangle(
        (cx_start, y_pos - BOX_HEIGHT / 2),
        issue_width, BOX_HEIGHT,
        linewidth=0, edgecolor='none', facecolor=colors['face'], zorder=4, alpha=ISSUE_ALPHA
    )
    ax.add_patch(issue_rect)

    if hatch is not None:
        hatch_rect = patches.Rectangle(
            (cx_start, y_pos - BOX_HEIGHT / 2),
            c_width, BOX_HEIGHT,
            linewidth=0, edgecolor='#d8d8d8', facecolor='none', hatch=hatch, zorder=4.5, alpha=0.3
        )
        ax.add_patch(hatch_rect)

    ax.text(
        cx_start + c_width / 2, y_pos, label,
        color='white', ha='center', va='center', fontsize=8, zorder=5, fontweight='bold', alpha=0.85
    )


def draw_overlap_schedule(schedule_m, ii, schedule_l, output_filename, op_wgid=None, active_wgids=None):
    op_y_pos, y_ticks_positions, y_ticks_labels, unit_boundaries, total_rows = build_y_layout(schedule_m)
    iteration_configs = make_iteration_configs(ii)
    plot_items, sorted_events, time_to_coord = build_overlap_items(
        schedule_m, ii, schedule_l, iteration_configs
    )

    fig, ax = plt.subplots(figsize=(26, max(10, total_rows * 0.34)), dpi=150)
    setup_dark_axis(fig, ax)

    draw_hatches = op_wgid is not None
    for op_key, start, end, issue_end, stage in plot_items:
        op, _ = parse_op_key(op_key)
        colors = ITERATION_COLOR_PALETTE[stage]
        wgid = get_warpgroup(op_wgid, op_key) if draw_hatches else None
        hatch = WG_HATCHES.get(wgid, '---') if wgid is not None else None
        label = f"{format_op_key(op_key)} {op_desc[op]} [{STAGE_LABELS[stage]}]"
        draw_schedule_block(
            ax,
            time_to_coord[start],
            time_to_coord[end],
            time_to_coord[issue_end],
            op_y_pos[op_key],
            colors,
            label,
            hatch=hatch,
        )

    draw_unit_guides(ax, len(sorted_events), unit_boundaries, y_ticks_positions, y_ticks_labels)

    important_times = {0, ii, ii * 2, ii * 3, ii * 2 + schedule_l}
    xtick_positions = choose_xticks(sorted_events, important_times)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(sorted_events[idx]) for idx in xtick_positions], fontsize=8, rotation=0)
    ax.tick_params(axis='x', pad=8, length=4)

    bottom_y = -1.0
    ax.spines['bottom'].set_position(('data', bottom_y))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    xmin, xmax = -0.5, len(sorted_events) - 0.5
    ax.set_xlim(xmin, xmax + 1.5)
    ax.set_ylim(bottom_y - 3.2, total_rows - 0.2)

    axis_iteration_start_idx = time_to_coord[ii * 2]
    axis_iteration_end_idx = time_to_coord[ii * 3]
    axis_edge_color = ITERATION_COLOR_PALETTE[2]['edge']
    range_arrow_y = bottom_y - 1.0
    ax.annotate('', xy=(axis_iteration_start_idx, range_arrow_y), xytext=(axis_iteration_end_idx, range_arrow_y), arrowprops=dict(arrowstyle="<->", color=axis_edge_color, lw=2))
    ax.text((axis_iteration_start_idx + axis_iteration_end_idx) / 2, bottom_y - 0.9, f"Iter I+1: II = {ii}", color=axis_edge_color, ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

    latency_idx = time_to_coord[ii * 2 + schedule_l]
    latency_arrow_y = bottom_y - 2.1
    ax.annotate('', xy=(axis_iteration_start_idx, latency_arrow_y), xytext=(latency_idx, latency_arrow_y), arrowprops=dict(arrowstyle="<->", color='#aaaaaa', lw=2))
    ax.text((axis_iteration_start_idx + latency_idx) / 2, bottom_y - 2.0, f"Iter I+1: L = {schedule_l}", color='#aaaaaa', ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

    ax.annotate('', xy=(xmax + 1.0, bottom_y), xytext=(xmin, bottom_y), arrowprops=dict(arrowstyle="->", color='white', lw=1.5))
    ax.text(xmax + 1.0, bottom_y + 0.18, 'Timeline (Hardware Cycle)', color='white', ha='right', va='bottom', fontsize=10)

    legend_patches = [
        patches.Patch(facecolor=ITERATION_COLOR_PALETTE[1]['face'], edgecolor=ITERATION_COLOR_PALETTE[1]['edge'], label="Iteration I-1 [i-1] Blocks"),
        patches.Patch(facecolor=ITERATION_COLOR_PALETTE[0]['face'], edgecolor=ITERATION_COLOR_PALETTE[0]['edge'], label="Iteration I [i] Blocks"),
        patches.Patch(facecolor=ITERATION_COLOR_PALETTE[2]['face'], edgecolor=ITERATION_COLOR_PALETTE[2]['edge'], label="Iteration I+1 [i+1] Blocks"),
        patches.Patch(facecolor='#666666', edgecolor='none', alpha=ISSUE_ALPHA, label='Dark = issue; light = execution'),
    ]
    iteration_legend = ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(0.82, 0.98),
        bbox_transform=fig.transFigure,
        facecolor='#1A1A1A',
        edgecolor='#444444',
        fontsize=10,
    )
    ax.add_artist(iteration_legend)

    if draw_hatches:
        wg_legend_patches = [
            patches.Patch(
                facecolor='#555555',
                edgecolor='#d8d8d8',
                hatch=WG_HATCHES.get(wgid, '---'),
                label=f"WG{wgid}: warpId // 4 == {wgid}",
            )
            for wgid in active_wgids
        ]
        ax.legend(
            handles=wg_legend_patches,
            loc='upper left',
            bbox_to_anchor=(0.82, 0.74),
            bbox_transform=fig.transFigure,
            facecolor='#1A1A1A',
            edgecolor='#444444',
            fontsize=10,
        )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"模调度 overlap 图已成功导出: {output_filename}")


def iter_label(distance):
    if distance == 0:
        return 'i'
    return f"i-{distance}"


def draw_single_modulo_window(schedule_m, ii, schedule_l, output_filename, op_wgid=None, active_wgids=None):
    op_y_pos, y_ticks_positions, y_ticks_labels, unit_boundaries, total_rows = build_y_layout(schedule_m)
    single_items = []
    single_events = {0, ii, schedule_l}
    for op_key, start_time in schedule_m.items():
        iter_distance = start_time // ii
        slot_start = start_time % ii
        slot_issue_end = slot_start + duration[op_key]
        slot_end = slot_start + latencies[op_key]
        single_items.append((op_key, slot_start, slot_end, slot_issue_end, iter_distance))
        single_events.update((slot_start, slot_issue_end, slot_end))

    single_items.sort(key=lambda item: (item[1], item[4], op_key_sort_key(item[0])))
    single_events = sorted(single_events)
    single_time_to_coord = {t: i for i, t in enumerate(single_events)}

    single_color_palette = {
        0: {'face': '#1f4e5b', 'edge': '#3a889e'},
        1: {'face': '#5c3d2e', 'edge': '#ba7a5f'},
        2: {'face': '#3f4f2f', 'edge': '#8aaa5e'},
    }
    fallback_colors = [
        {'face': '#4b3f72', 'edge': '#9d87d2'},
        {'face': '#554b2f', 'edge': '#c0a45d'},
    ]

    fig_single, ax_single = plt.subplots(figsize=(26, max(10, total_rows * 0.34)), dpi=150)
    setup_dark_axis(fig_single, ax_single)

    draw_hatches = op_wgid is not None
    for op_key, slot_start, slot_end, slot_issue_end, iter_distance in single_items:
        op, _ = parse_op_key(op_key)
        colors = single_color_palette.get(
            iter_distance,
            fallback_colors[iter_distance % len(fallback_colors)],
        )
        wgid = get_warpgroup(op_wgid, op_key) if draw_hatches else None
        hatch = WG_HATCHES.get(wgid, '---') if wgid is not None else None
        label = f"{format_op_key(op_key)} {op_desc[op]} [{iter_label(iter_distance)}]"
        draw_schedule_block(
            ax_single,
            single_time_to_coord[slot_start],
            single_time_to_coord[slot_end],
            single_time_to_coord[slot_issue_end],
            op_y_pos[op_key],
            colors,
            label,
            hatch=hatch,
        )

    draw_unit_guides(ax_single, len(single_events), unit_boundaries, y_ticks_positions, y_ticks_labels)

    single_xtick_positions = sorted({
        idx
        for idx, t in enumerate(single_events)
        if t in {0, ii, schedule_l} or idx % max(1, int(np.ceil(len(single_events) / 34))) == 0
    })
    ax_single.set_xticks(single_xtick_positions)
    ax_single.set_xticklabels([str(single_events[idx]) for idx in single_xtick_positions], fontsize=8, rotation=0)
    ax_single.tick_params(axis='x', pad=8, length=4)

    single_bottom_y = -1.0
    ax_single.spines['bottom'].set_position(('data', single_bottom_y))
    ax_single.spines['top'].set_visible(False)
    ax_single.spines['right'].set_visible(False)
    ax_single.spines['left'].set_visible(False)
    ax_single.set_xlim(-0.5, len(single_events) - 0.5)
    ax_single.set_ylim(single_bottom_y - 2.4, total_rows - 0.2)

    ii_start_idx = single_time_to_coord[0]
    ii_end_idx = single_time_to_coord[ii]
    ax_single.annotate('', xy=(ii_start_idx, single_bottom_y - 1.0), xytext=(ii_end_idx, single_bottom_y - 1.0), arrowprops=dict(arrowstyle="<->", color='#3a889e', lw=2))
    ax_single.text((ii_start_idx + ii_end_idx) / 2, single_bottom_y - 0.9, f"Single modulo window: II = {ii}", color='#3a889e', ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

    latency_idx = single_time_to_coord[schedule_l]
    ax_single.annotate('', xy=(ii_start_idx, single_bottom_y - 2.0), xytext=(latency_idx, single_bottom_y - 2.0), arrowprops=dict(arrowstyle="<->", color='#aaaaaa', lw=2))
    ax_single.text((ii_start_idx + latency_idx) / 2, single_bottom_y - 1.9, f"Logical iteration latency L = {schedule_l}", color='#aaaaaa', ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

    ax_single.annotate('', xy=(len(single_events) - 0.5, single_bottom_y), xytext=(-0.5, single_bottom_y), arrowprops=dict(arrowstyle="->", color='white', lw=1.5))
    ax_single.text(len(single_events) - 0.5, single_bottom_y + 0.18, 'Modulo slot / unfolded latency markers', color='white', ha='right', va='bottom', fontsize=10)

    iteration_distances = sorted({item[4] for item in single_items})
    single_iteration_legend = [
        patches.Patch(
            facecolor=single_color_palette.get(distance, fallback_colors[distance % len(fallback_colors)])['face'],
            edgecolor=single_color_palette.get(distance, fallback_colors[distance % len(fallback_colors)])['edge'],
            label=f"ops from iteration {iter_label(distance)}",
        )
        for distance in iteration_distances
    ]
    single_iteration_legend.append(
        patches.Patch(facecolor='#666666', edgecolor='none', alpha=ISSUE_ALPHA, label='Dark = issue; light = execution')
    )
    iter_legend = ax_single.legend(
        handles=single_iteration_legend,
        loc='upper left',
        bbox_to_anchor=(0.82, 0.98),
        bbox_transform=fig_single.transFigure,
        facecolor='#1A1A1A',
        edgecolor='#444444',
        fontsize=10,
    )
    ax_single.add_artist(iter_legend)

    if draw_hatches:
        single_wg_legend = [
            patches.Patch(
                facecolor='#555555',
                edgecolor='#d8d8d8',
                hatch=WG_HATCHES.get(wgid, '---'),
                label=f"WG{wgid}: warpId // 4 == {wgid}",
            )
            for wgid in active_wgids
        ]
        ax_single.legend(
            handles=single_wg_legend,
            loc='upper left',
            bbox_to_anchor=(0.82, 0.74),
            bbox_transform=fig_single.transFigure,
            facecolor='#1A1A1A',
            edgecolor='#444444',
            fontsize=10,
        )

    fig_single.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(output_filename, dpi=300, facecolor=fig_single.get_facecolor(), edgecolor='none')
    plt.close(fig_single)
    print(f"单个 II 窗口内的 op 启动顺序图已成功导出: {output_filename}")


def aggregate_lifetimes_by_buffer(variable_lifetime_items):
    """Aggregate copy-level lifetimes into one interval per buffer and iteration."""
    if isinstance(variable_lifetime_items, dict):
        buffer_items = variable_lifetime_items.values()
    else:
        buffer_items = variable_lifetime_items

    aggregated_items = []
    for buffer_info in buffer_items:
        copies_by_iter = {}
        for copy in buffer_info.get('copies', []):
            iter_offset = copy['iter_offset']
            merged = copies_by_iter.setdefault(
                iter_offset,
                {
                    'name': buffer_info.get('name', buffer_info.get('buffer')),
                    'buffer': buffer_info.get('buffer', buffer_info.get('name')),
                    'storage': buffer_info.get('storage', 'UNKNOWN'),
                    'footprint_bytes': buffer_info.get('footprint_bytes', 0),
                    'lifetime': buffer_info.get('lifetime', 'unknown'),
                    'producers': set(buffer_info.get('producers', [])),
                    'iter_offset': iter_offset,
                    'live_start': copy['live_start'],
                    'live_end': copy['live_end'],
                    'live_end_exclusive': copy.get('live_end_exclusive', copy['live_end']),
                    'consumers': [],
                },
            )
            merged['live_start'] = min(merged['live_start'], copy['live_start'])
            merged['live_end'] = max(merged['live_end'], copy['live_end'])
            merged['live_end_exclusive'] = max(
                merged['live_end_exclusive'],
                copy.get('live_end_exclusive', copy['live_end']),
            )
            merged['producers'].add(copy.get('producer'))
            merged['consumers'].extend(copy.get('consumers', []))

        for item in copies_by_iter.values():
            item['producers'] = sorted(producer for producer in item['producers'] if producer)
            aggregated_items.append(item)

    return aggregated_items


def build_lifetime_layout(lifetime_items):
    """Build one y row per buffer and keep all iteration intervals on that row."""
    buffer_names = sorted({item['buffer'] for item in lifetime_items})
    y_by_buffer = {
        name: idx * LIFETIME_ROW_GAP
        for idx, name in enumerate(reversed(buffer_names))
    }
    y_ticks = [y_by_buffer[name] for name in buffer_names]
    y_labels = list(buffer_names)
    total_height = max(y_by_buffer.values(), default=0) + LIFETIME_ROW_GAP
    return y_by_buffer, y_ticks, y_labels, total_height


def lifetime_stage_label(iter_offset):
    if iter_offset == -1:
        return 'i-1'
    if iter_offset == 0:
        return 'i'
    if iter_offset == 1:
        return 'i+1'
    if iter_offset > 0:
        return f"i+{iter_offset}"
    return f"i{iter_offset}"


def build_lifetime_events(lifetime_items, ii, schedule_l):
    events = {0, ii, ii * 2, ii * 3, schedule_l, ii + schedule_l, ii * 2 + schedule_l}
    for item in lifetime_items:
        events.add(item['live_start'])
        events.add(item['live_end'])
        events.add(item['live_end_exclusive'])
        for consumer in item.get('consumers', []):
            events.add(consumer['consume_time'])
    sorted_events = sorted(events)
    return sorted_events, {t: idx for idx, t in enumerate(sorted_events)}


def lifetime_iter_hatch(iter_offset):
    return '\\\\\\' if iter_offset > 0 else 'xxx'


def lifetime_iter_colors(iter_offset):
    fallback_colors = [
        {'face': '#4b3f72', 'edge': '#9d87d2'},
        {'face': '#554b2f', 'edge': '#c0a45d'},
    ]
    return LIFETIME_ITER_COLORS.get(
        iter_offset,
        fallback_colors[abs(iter_offset) % len(fallback_colors)],
    )


def lifetime_storage_hatch(storage):
    return LIFETIME_STORAGE_HATCHES.get(storage, '\\\\\\')


def lifetime_iter_y_offset(iter_offset):
    return iter_offset * LIFETIME_ITER_Y_STEP


def draw_lifetime_block(ax, cx_start, cx_end, y_pos, colors, label, hatch=None, alpha=0.82):
    c_width = max(cx_end - cx_start, 0.25)
    rect = patches.Rectangle(
        (cx_start, y_pos - LIFETIME_BOX_HEIGHT / 2),
        c_width,
        LIFETIME_BOX_HEIGHT,
        linewidth=1.4,
        edgecolor=colors['edge'],
        facecolor=colors['face'],
        zorder=3,
        alpha=alpha,
    )
    ax.add_patch(rect)
    if hatch:
        hatch_rect = patches.Rectangle(
            (cx_start, y_pos - LIFETIME_BOX_HEIGHT / 2),
            c_width,
            LIFETIME_BOX_HEIGHT,
            linewidth=0,
            edgecolor='#f0f0f0',
            facecolor='none',
            hatch=hatch,
            zorder=4,
            alpha=0.45,
        )
        ax.add_patch(hatch_rect)
    ax.text(
        cx_start + c_width / 2,
        y_pos,
        label,
        color='white',
        ha='center',
        va='center',
        fontsize=7,
        zorder=5,
        fontweight='bold',
        alpha=0.9,
    )


def draw_variable_lifetimes(lifetime_items, ii, schedule_l, output_filename):
    lifetime_items = aggregate_lifetimes_by_buffer(lifetime_items)
    y_by_name, y_ticks, y_labels, total_height = build_lifetime_layout(lifetime_items)
    sorted_events, time_to_coord = build_lifetime_events(lifetime_items, ii, schedule_l)

    fig, ax = plt.subplots(figsize=(26, max(10, total_height * 0.52)), dpi=150)
    setup_dark_axis(fig, ax)

    for idx in range(len(sorted_events)):
        ax.axvline(x=idx, color='#2c2c2c', linestyle=':', linewidth=0.9, zorder=1)

    iteration_guides = [
        (-ii, 'i-1 start', '#ba7a5f'),
        (0, 'i start', '#3a889e'),
        (ii, 'i+1 start', '#8aaa5e'),
        (ii * 2, 'i+2 start', '#777777'),
    ]
    for t, label, color in iteration_guides:
        if t not in time_to_coord:
            continue
        x = time_to_coord[t]
        ax.axvline(x=x, color=color, linestyle='-', linewidth=1.6, alpha=0.9, zorder=2)
        ax.text(x + 0.15, total_height - 0.2, label, color=color, ha='left', va='top', fontsize=9)

    for item in sorted(lifetime_items, key=lambda x: (x['buffer'], x['iter_offset'], x['live_start'])):
        y_pos = y_by_name[item['buffer']] + lifetime_iter_y_offset(item['iter_offset'])
        colors = lifetime_iter_colors(item['iter_offset'])
        cx_start = time_to_coord[item['live_start']]
        cx_end = time_to_coord[item['live_end_exclusive']]
        label = (
            f"{lifetime_stage_label(item['iter_offset'])} "
            f"{item['storage']} {item['footprint_bytes']}B "
            f"{item['live_start']}..{item['live_end']}"
        )
        draw_lifetime_block(
            ax,
            cx_start,
            cx_end,
            y_pos,
            colors,
            label,
            hatch=lifetime_storage_hatch(item['storage']),
        )

        for consumer in item.get('consumers', []):
            consume_x = time_to_coord[consumer['consume_time']]
            ax.vlines(
                consume_x,
                y_pos - LIFETIME_BOX_HEIGHT / 2,
                y_pos + LIFETIME_BOX_HEIGHT / 2,
                color='#f4d35e',
                linewidth=1.5,
                zorder=6,
            )
            ax.text(
                consume_x,
                y_pos + LIFETIME_BOX_HEIGHT / 2 + 0.05,
                consumer['consumer'],
                color='#f4d35e',
                ha='center',
                va='bottom',
                fontsize=6,
                zorder=7,
            )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)

    important_times = {0, ii, ii * 2, ii * 3, schedule_l, ii + schedule_l, ii * 2 + schedule_l}
    xtick_positions = choose_xticks(sorted_events, important_times, max_xtick_labels=42, min_tick_gap=2)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(sorted_events[idx]) for idx in xtick_positions], fontsize=8, rotation=0)
    ax.tick_params(axis='x', pad=8, length=4)

    bottom_y = -1.0
    ax.spines['bottom'].set_position(('data', bottom_y))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-0.5, len(sorted_events) - 0.5)
    ax.set_ylim(bottom_y - 1.5, total_height - 0.2)

    ax.annotate('', xy=(len(sorted_events) - 0.5, bottom_y), xytext=(-0.5, bottom_y), arrowprops=dict(arrowstyle="->", color='white', lw=1.5))
    ax.text(len(sorted_events) - 0.5, bottom_y + 0.18, 'Lifetime timeline (Hardware Cycle)', color='white', ha='right', va='bottom', fontsize=10)

    iter_offsets = sorted({item['iter_offset'] for item in lifetime_items})
    iter_legend_patches = [
        patches.Patch(
            facecolor=lifetime_iter_colors(iter_offset)['face'],
            edgecolor=lifetime_iter_colors(iter_offset)['edge'],
            label=f"{lifetime_stage_label(iter_offset)} lifetime",
        )
        for iter_offset in iter_offsets
    ]
    iter_legend_patches.append(
        patches.Patch(facecolor='#f4d35e', edgecolor='#f4d35e', label='consumer consume_time marker')
    )
    iter_legend = ax.legend(handles=iter_legend_patches, loc='upper right', facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)
    ax.add_artist(iter_legend)

    storage_legend_patches = [
        patches.Patch(
            facecolor='#555555',
            edgecolor='#f0f0f0',
            hatch=lifetime_storage_hatch(storage),
            label=f"{storage} storage",
        )
        for storage in sorted({item['storage'] for item in lifetime_items})
    ]
    ax.legend(
        handles=storage_legend_patches,
        loc='upper right',
        bbox_to_anchor=(1.0, 0.84),
        facecolor='#1A1A1A',
        edgecolor='#444444',
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"按 buffer 聚合的变量 lifetime 甘特图已成功导出: {output_filename}")


def main():
    plt.style.use('dark_background')
    schedule_source = expanded_M or optimized_M
    fine_optimized_M = build_fine_grained_schedule(
        schedule_source,
        latencies,
        duration,
        allow_parent_fallback=not bool(expanded_M),
    )
    op_wgid, active_wgids, wgid_ops = build_warpgroup_info(warp_assign)
    print(wgid_ops)

    draw_overlap_schedule(
        fine_optimized_M,
        base_I,
        optimized_L,
        'fine_grained_modulo_scheduling_same_line_with_desc.png',
        op_wgid=op_wgid,
        active_wgids=active_wgids,
    )
    draw_single_modulo_window(
        fine_optimized_M,
        base_I,
        optimized_L,
        'fine_grained_single_iter_ops.png',
        op_wgid=op_wgid,
        active_wgids=active_wgids,
    )

    # draw_variable_lifetimes(variable_lifetimes, base_I, optimized_L, 'var_life.png')


if __name__ == '__main__':
    main()
