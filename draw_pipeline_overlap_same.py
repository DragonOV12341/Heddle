import os
import math

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
warp_assign={'s0': 7, 's2': 10, 's3': 9, 's4': 11, 's5': 8, 's6': 10, 's7': 10, 's8': 8, 's9': 8, 's10': 11, 's11': 8, 's12': 11, 's13': 9, 's14': 8, 's15': 8, 's16': 11, 's17': 8, 's18': 11, 's19': 8, 's20': 8, 's21': 7, 's23': 8}
optimized_L=1250
base_I=1161
optimized_M={0: 0, 2: 0, 3: 5, 4: 0, 5: 282, 6: 818, 7: 822, 8: 5, 9: 828, 10: 834, 11: 840, 12: 818, 13: 828, 14: 864, 15: 900, 16: 1174, 17: 874, 18: 874, 19: 1178, 20: 1174, 21: 0, 23: 1186}
variable_lifetimes={'Ks': {'name': 'Ks', 'storage': 'SMEM', 'buffer': 'Ks', 'footprint_bytes': 16384, 'lifetime': 'dead_on_entry', 'producers': ['s0'], 'copies': [{'producer': 's0', 'producer_warp': 7, 'iter_offset': -1, 'live_start': -1161, 'live_end': -880, 'live_end_exclusive': -879, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': -879}]}, {'producer': 's0', 'producer_warp': 7, 'iter_offset': 0, 'live_start': 0, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 282}]}, {'producer': 's0', 'producer_warp': 7, 'iter_offset': 1, 'live_start': 1161, 'live_end': 1442, 'live_end_exclusive': 1443, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 1443}]}]}, 'smp': {'name': 'smp', 'storage': 'RMEM', 'buffer': 'smp', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s2'], 'copies': [{'producer': 's2', 'producer_warp': 10, 'iter_offset': -1, 'live_start': -1161, 'live_end': -340, 'live_end_exclusive': -339, 'consumers': [{'consumer': 's3', 'distance': 0, 'consume_time': -1156}, {'consumer': 's7', 'distance': 0, 'consume_time': -339}, {'consumer': 's8', 'distance': 0, 'consume_time': -1156}]}, {'producer': 's2', 'producer_warp': 10, 'iter_offset': 0, 'live_start': 0, 'live_end': 821, 'live_end_exclusive': 822, 'consumers': [{'consumer': 's3', 'distance': 0, 'consume_time': 5}, {'consumer': 's7', 'distance': 0, 'consume_time': 822}, {'consumer': 's8', 'distance': 0, 'consume_time': 5}]}, {'producer': 's2', 'producer_warp': 10, 'iter_offset': 1, 'live_start': 1161, 'live_end': 1982, 'live_end_exclusive': 1983, 'consumers': [{'consumer': 's3', 'distance': 0, 'consume_time': 1166}, {'consumer': 's7', 'distance': 0, 'consume_time': 1983}, {'consumer': 's8', 'distance': 0, 'consume_time': 1166}]}]}, 'sm': {'name': 'sm', 'storage': 'RMEM', 'buffer': 'sm', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s3', 's6', 's7'], 'copies': [{'producer': 's3', 'producer_warp': 9, 'iter_offset': -1, 'live_start': -1156, 'live_end': -344, 'live_end_exclusive': -343, 'consumers': [{'consumer': 's6', 'distance': 0, 'consume_time': -343}]}, {'producer': 's3', 'producer_warp': 9, 'iter_offset': 0, 'live_start': 5, 'live_end': 817, 'live_end_exclusive': 818, 'consumers': [{'consumer': 's6', 'distance': 0, 'consume_time': 818}]}, {'producer': 's3', 'producer_warp': 9, 'iter_offset': 1, 'live_start': 1166, 'live_end': 1978, 'live_end_exclusive': 1979, 'consumers': [{'consumer': 's6', 'distance': 0, 'consume_time': 1979}]}, {'producer': 's6', 'producer_warp': 10, 'iter_offset': -1, 'live_start': -343, 'live_end': 817, 'live_end_exclusive': 818, 'consumers': [{'consumer': 's6', 'distance': 1, 'consume_time': 818}, {'consumer': 's7', 'distance': 0, 'consume_time': -339}]}, {'producer': 's6', 'producer_warp': 10, 'iter_offset': 0, 'live_start': 818, 'live_end': 1978, 'live_end_exclusive': 1979, 'consumers': [{'consumer': 's6', 'distance': 1, 'consume_time': 1979}, {'consumer': 's7', 'distance': 0, 'consume_time': 822}]}, {'producer': 's6', 'producer_warp': 10, 'iter_offset': 1, 'live_start': 1979, 'live_end': 3139, 'live_end_exclusive': 3140, 'consumers': [{'consumer': 's6', 'distance': 1, 'consume_time': 3140}, {'consumer': 's7', 'distance': 0, 'consume_time': 1983}]}, {'producer': 's7', 'producer_warp': 10, 'iter_offset': -1, 'live_start': -339, 'live_end': 821, 'live_end_exclusive': 822, 'consumers': [{'consumer': 's7', 'distance': 1, 'consume_time': 822}, {'consumer': 's9', 'distance': 0, 'consume_time': -333}, {'consumer': 's13', 'distance': 0, 'consume_time': -333}]}, {'producer': 's7', 'producer_warp': 10, 'iter_offset': 0, 'live_start': 822, 'live_end': 1982, 'live_end_exclusive': 1983, 'consumers': [{'consumer': 's7', 'distance': 1, 'consume_time': 1983}, {'consumer': 's9', 'distance': 0, 'consume_time': 828}, {'consumer': 's13', 'distance': 0, 'consume_time': 828}]}, {'producer': 's7', 'producer_warp': 10, 'iter_offset': 1, 'live_start': 1983, 'live_end': 3143, 'live_end_exclusive': 3144, 'consumers': [{'consumer': 's7', 'distance': 1, 'consume_time': 3144}, {'consumer': 's9', 'distance': 0, 'consume_time': 1989}, {'consumer': 's13', 'distance': 0, 'consume_time': 1989}]}]}, 'acc_s': {'name': 'acc_s', 'storage': 'RMEM', 'buffer': 'acc_s', 'footprint_bytes': 128, 'lifetime': 'dead_on_entry', 'producers': ['s4', 's5', 's15'], 'copies': [{'producer': 's4', 'producer_warp': 11, 'iter_offset': -1, 'live_start': -1161, 'live_end': -880, 'live_end_exclusive': -879, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': -879}]}, {'producer': 's4', 'producer_warp': 11, 'iter_offset': 0, 'live_start': 0, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 282}]}, {'producer': 's4', 'producer_warp': 11, 'iter_offset': 1, 'live_start': 1161, 'live_end': 1442, 'live_end_exclusive': 1443, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 1443}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -879, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 282}, {'consumer': 's6', 'distance': 0, 'consume_time': -343}, {'consumer': 's12', 'distance': 0, 'consume_time': -343}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 282, 'live_end': 1442, 'live_end_exclusive': 1443, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 1443}, {'consumer': 's6', 'distance': 0, 'consume_time': 818}, {'consumer': 's12', 'distance': 0, 'consume_time': 818}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 1443, 'live_end': 2603, 'live_end_exclusive': 2604, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 2604}, {'consumer': 's6', 'distance': 0, 'consume_time': 1979}, {'consumer': 's12', 'distance': 0, 'consume_time': 1979}]}, {'producer': 's15', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -261, 'live_end': 12, 'live_end_exclusive': 13, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 13}, {'consumer': 's20', 'distance': 0, 'consume_time': 13}]}, {'producer': 's15', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 900, 'live_end': 1173, 'live_end_exclusive': 1174, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 1174}, {'consumer': 's20', 'distance': 0, 'consume_time': 1174}]}, {'producer': 's15', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2061, 'live_end': 2334, 'live_end_exclusive': 2335, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 2335}, {'consumer': 's20', 'distance': 0, 'consume_time': 2335}]}]}, 'desc_a': {'name': 'desc_a', 'storage': 'RMEM', 'buffer': 'desc_a', 'footprint_bytes': 2, 'lifetime': 'dead_on_entry', 'producers': ['s5'], 'copies': [{'producer': 's5', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -879, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 282}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 282, 'live_end': 1442, 'live_end_exclusive': 1443, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 1443}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 1443, 'live_end': 2603, 'live_end_exclusive': 2604, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 2604}]}]}, 'desc_b': {'name': 'desc_b', 'storage': 'RMEM', 'buffer': 'desc_b', 'footprint_bytes': 2, 'lifetime': 'dead_on_entry', 'producers': ['s5', 's23'], 'copies': [{'producer': 's5', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -879, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 282}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 282, 'live_end': 1442, 'live_end_exclusive': 1443, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 1443}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 1443, 'live_end': 2603, 'live_end_exclusive': 2604, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 2604}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 25, 'live_end': 1185, 'live_end_exclusive': 1186, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 1186}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 1186, 'live_end': 2346, 'live_end_exclusive': 2347, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 2347}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2347, 'live_end': 3507, 'live_end_exclusive': 3508, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 3508}]}]}, '_tmp_0': {'name': '_tmp_0', 'storage': 'RMEM', 'buffer': '_tmp_0', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s8'], 'copies': [{'producer': 's8', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -1156, 'live_end': -328, 'live_end_exclusive': -327, 'consumers': [{'consumer': 's10', 'distance': 0, 'consume_time': -327}]}, {'producer': 's8', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 5, 'live_end': 833, 'live_end_exclusive': 834, 'consumers': [{'consumer': 's10', 'distance': 0, 'consume_time': 834}]}, {'producer': 's8', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 1166, 'live_end': 1994, 'live_end_exclusive': 1995, 'consumers': [{'consumer': 's10', 'distance': 0, 'consume_time': 1995}]}]}, '_tmp_1': {'name': '_tmp_1', 'storage': 'RMEM', 'buffer': '_tmp_1', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s9'], 'copies': [{'producer': 's9', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -333, 'live_end': -328, 'live_end_exclusive': -327, 'consumers': [{'consumer': 's10', 'distance': 0, 'consume_time': -327}]}, {'producer': 's9', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 828, 'live_end': 833, 'live_end_exclusive': 834, 'consumers': [{'consumer': 's10', 'distance': 0, 'consume_time': 834}]}, {'producer': 's9', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 1989, 'live_end': 1994, 'live_end_exclusive': 1995, 'consumers': [{'consumer': 's10', 'distance': 0, 'consume_time': 1995}]}]}, '_tmp_2': {'name': '_tmp_2', 'storage': 'RMEM', 'buffer': '_tmp_2', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s10'], 'copies': [{'producer': 's10', 'producer_warp': 11, 'iter_offset': -1, 'live_start': -327, 'live_end': -322, 'live_end_exclusive': -321, 'consumers': [{'consumer': 's11', 'distance': 0, 'consume_time': -321}]}, {'producer': 's10', 'producer_warp': 11, 'iter_offset': 0, 'live_start': 834, 'live_end': 839, 'live_end_exclusive': 840, 'consumers': [{'consumer': 's11', 'distance': 0, 'consume_time': 840}]}, {'producer': 's10', 'producer_warp': 11, 'iter_offset': 1, 'live_start': 1995, 'live_end': 2000, 'live_end_exclusive': 2001, 'consumers': [{'consumer': 's11', 'distance': 0, 'consume_time': 2001}]}]}, 'ss': {'name': 'ss', 'storage': 'RMEM', 'buffer': 'ss', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s11'], 'copies': [{'producer': 's11', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -321, 'live_end': -288, 'live_end_exclusive': -287, 'consumers': [{'consumer': 's17', 'distance': 0, 'consume_time': -287}, {'consumer': 's18', 'distance': 0, 'consume_time': -287}]}, {'producer': 's11', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 840, 'live_end': 873, 'live_end_exclusive': 874, 'consumers': [{'consumer': 's17', 'distance': 0, 'consume_time': 874}, {'consumer': 's18', 'distance': 0, 'consume_time': 874}]}, {'producer': 's11', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2001, 'live_end': 2034, 'live_end_exclusive': 2035, 'consumers': [{'consumer': 's17', 'distance': 0, 'consume_time': 2035}, {'consumer': 's18', 'distance': 0, 'consume_time': 2035}]}]}, '_tmp_3': {'name': '_tmp_3', 'storage': 'RMEM', 'buffer': '_tmp_3', 'footprint_bytes': 128, 'lifetime': 'dead_on_entry', 'producers': ['s12'], 'copies': [{'producer': 's12', 'producer_warp': 11, 'iter_offset': -1, 'live_start': -343, 'live_end': -298, 'live_end_exclusive': -297, 'consumers': [{'consumer': 's14', 'distance': 0, 'consume_time': -297}]}, {'producer': 's12', 'producer_warp': 11, 'iter_offset': 0, 'live_start': 818, 'live_end': 863, 'live_end_exclusive': 864, 'consumers': [{'consumer': 's14', 'distance': 0, 'consume_time': 864}]}, {'producer': 's12', 'producer_warp': 11, 'iter_offset': 1, 'live_start': 1979, 'live_end': 2024, 'live_end_exclusive': 2025, 'consumers': [{'consumer': 's14', 'distance': 0, 'consume_time': 2025}]}]}, '_tmp_4': {'name': '_tmp_4', 'storage': 'RMEM', 'buffer': '_tmp_4', 'footprint_bytes': 128, 'lifetime': 'dead_on_entry', 'producers': ['s13'], 'copies': [{'producer': 's13', 'producer_warp': 9, 'iter_offset': -1, 'live_start': -333, 'live_end': -298, 'live_end_exclusive': -297, 'consumers': [{'consumer': 's14', 'distance': 0, 'consume_time': -297}]}, {'producer': 's13', 'producer_warp': 9, 'iter_offset': 0, 'live_start': 828, 'live_end': 863, 'live_end_exclusive': 864, 'consumers': [{'consumer': 's14', 'distance': 0, 'consume_time': 864}]}, {'producer': 's13', 'producer_warp': 9, 'iter_offset': 1, 'live_start': 1989, 'live_end': 2024, 'live_end_exclusive': 2025, 'consumers': [{'consumer': 's14', 'distance': 0, 'consume_time': 2025}]}]}, '_tmp_5': {'name': '_tmp_5', 'storage': 'RMEM', 'buffer': '_tmp_5', 'footprint_bytes': 128, 'lifetime': 'dead_on_entry', 'producers': ['s14'], 'copies': [{'producer': 's14', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -297, 'live_end': -262, 'live_end_exclusive': -261, 'consumers': [{'consumer': 's15', 'distance': 0, 'consume_time': -261}]}, {'producer': 's14', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 864, 'live_end': 899, 'live_end_exclusive': 900, 'consumers': [{'consumer': 's15', 'distance': 0, 'consume_time': 900}]}, {'producer': 's14', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2025, 'live_end': 2060, 'live_end_exclusive': 2061, 'consumers': [{'consumer': 's15', 'distance': 0, 'consume_time': 2061}]}]}, 'ssum': {'name': 'ssum', 'storage': 'RMEM', 'buffer': 'ssum', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s16'], 'copies': [{'producer': 's16', 'producer_warp': 11, 'iter_offset': -1, 'live_start': 13, 'live_end': 1173, 'live_end_exclusive': 1174, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 1174}, {'consumer': 's19', 'distance': 0, 'consume_time': 17}]}, {'producer': 's16', 'producer_warp': 11, 'iter_offset': 0, 'live_start': 1174, 'live_end': 2334, 'live_end_exclusive': 2335, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 2335}, {'consumer': 's19', 'distance': 0, 'consume_time': 1178}]}, {'producer': 's16', 'producer_warp': 11, 'iter_offset': 1, 'live_start': 2335, 'live_end': 3495, 'live_end_exclusive': 3496, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 3496}, {'consumer': 's19', 'distance': 0, 'consume_time': 2339}]}]}, 'acc_o': {'name': 'acc_o', 'storage': 'RMEM', 'buffer': 'acc_o', 'footprint_bytes': 256, 'lifetime': 'dead_on_entry', 'producers': ['s17', 's23'], 'copies': [{'producer': 's17', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -287, 'live_end': 873, 'live_end_exclusive': 874, 'consumers': [{'consumer': 's17', 'distance': 1, 'consume_time': 874}, {'consumer': 's23', 'distance': 0, 'consume_time': 25}]}, {'producer': 's17', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 874, 'live_end': 2034, 'live_end_exclusive': 2035, 'consumers': [{'consumer': 's17', 'distance': 1, 'consume_time': 2035}, {'consumer': 's23', 'distance': 0, 'consume_time': 1186}]}, {'producer': 's17', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2035, 'live_end': 3195, 'live_end_exclusive': 3196, 'consumers': [{'consumer': 's17', 'distance': 1, 'consume_time': 3196}, {'consumer': 's23', 'distance': 0, 'consume_time': 2347}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 25, 'live_end': 1185, 'live_end_exclusive': 1186, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 1186}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 1186, 'live_end': 2346, 'live_end_exclusive': 2347, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 2347}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2347, 'live_end': 3507, 'live_end_exclusive': 3508, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 3508}]}]}, '_tmp_6': {'name': '_tmp_6', 'storage': 'RMEM', 'buffer': '_tmp_6', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s18'], 'copies': [{'producer': 's18', 'producer_warp': 11, 'iter_offset': -1, 'live_start': -287, 'live_end': 16, 'live_end_exclusive': 17, 'consumers': [{'consumer': 's19', 'distance': 0, 'consume_time': 17}]}, {'producer': 's18', 'producer_warp': 11, 'iter_offset': 0, 'live_start': 874, 'live_end': 1177, 'live_end_exclusive': 1178, 'consumers': [{'consumer': 's19', 'distance': 0, 'consume_time': 1178}]}, {'producer': 's18', 'producer_warp': 11, 'iter_offset': 1, 'live_start': 2035, 'live_end': 2338, 'live_end_exclusive': 2339, 'consumers': [{'consumer': 's19', 'distance': 0, 'consume_time': 2339}]}]}, 'ls': {'name': 'ls', 'storage': 'RMEM', 'buffer': 'ls', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s19'], 'copies': [{'producer': 's19', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 17, 'live_end': None, 'live_end_exclusive': 17, 'consumers': []}, {'producer': 's19', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 1178, 'live_end': None, 'live_end_exclusive': 1178, 'consumers': []}, {'producer': 's19', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2339, 'live_end': None, 'live_end_exclusive': 2339, 'consumers': []}]}, 'acc_s_c': {'name': 'acc_s_c', 'storage': 'RMEM', 'buffer': 'acc_s_c', 'footprint_bytes': 64, 'lifetime': 'dead_on_entry', 'producers': ['s20', 's23'], 'copies': [{'producer': 's20', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 13, 'live_end': 24, 'live_end_exclusive': 25, 'consumers': [{'consumer': 's23', 'distance': 0, 'consume_time': 25}]}, {'producer': 's20', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 1174, 'live_end': 1185, 'live_end_exclusive': 1186, 'consumers': [{'consumer': 's23', 'distance': 0, 'consume_time': 1186}]}, {'producer': 's20', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2335, 'live_end': 2346, 'live_end_exclusive': 2347, 'consumers': [{'consumer': 's23', 'distance': 0, 'consume_time': 2347}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 25, 'live_end': 1185, 'live_end_exclusive': 1186, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 1186}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 1186, 'live_end': 2346, 'live_end_exclusive': 2347, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 2347}]}, {'producer': 's23', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 2347, 'live_end': 3507, 'live_end_exclusive': 3508, 'consumers': [{'consumer': 's23', 'distance': 1, 'consume_time': 3508}]}]}, 'Vs': {'name': 'Vs', 'storage': 'SMEM', 'buffer': 'Vs', 'footprint_bytes': 16384, 'lifetime': 'dead_on_entry', 'producers': ['s21'], 'copies': [{'producer': 's21', 'producer_warp': 7, 'iter_offset': -1, 'live_start': -1161, 'live_end': 24, 'live_end_exclusive': 25, 'consumers': [{'consumer': 's23', 'distance': 0, 'consume_time': 25}]}, {'producer': 's21', 'producer_warp': 7, 'iter_offset': 0, 'live_start': 0, 'live_end': 1185, 'live_end_exclusive': 1186, 'consumers': [{'consumer': 's23', 'distance': 0, 'consume_time': 1186}]}, {'producer': 's21', 'producer_warp': 7, 'iter_offset': 1, 'live_start': 1161, 'live_end': 2346, 'live_end_exclusive': 2347, 'consumers': [{'consumer': 's23', 'distance': 0, 'consume_time': 2347}]}]}}

# naive mod sched
I = 1161
L = 1250
M = {0: 0, 2: 808, 3: 813, 4: 270, 5: 282, 6: 818, 7: 822, 8: 1072, 9: 1072, 10: 1078, 11: 1084, 12: 828, 13: 828, 14: 864, 15: 900, 16: 1244, 17: 1118, 18: 1242, 19: 1248, 20: 1174, 21: 904, 23: 1186}

# ========================================
# 执行+发射延迟
latencies={0: 282, 2: 5, 3: 5, 4: 12, 5: 536, 6: 4, 7: 6, 8: 6, 9: 6, 10: 6, 11: 34, 12: 36, 13: 36, 14: 36, 15: 274, 16: 4, 17: 68, 18: 6, 19: 6, 20: 12, 21: 282, 23: 304}
# 发射延迟
duration={0: 2, 2: 1, 3: 1, 4: 8, 5: 64, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 16, 12: 32, 13: 32, 14: 32, 15: 256, 16: 2, 17: 64, 18: 2, 19: 2, 20: 8, 21: 2, 23: 64}

# op_desc = {
#     0 : "tma_copy Ks" ,
#     1 : "wait Ks" ,
#     2 : "smp = sm" ,
#     3 : "clear sm" ,
#     4 : "clear acc_s" ,
#     5 : "acc_s = wgmma QK" ,
#     6 : "softmax: reduce_max(QK)" ,
#     7 : "softmax: update global max" ,
#     8 : "softmax: get scale" ,
#     9 : "softmax: exp" ,
#     10 : "softmax: sumexp" ,
#     11 : "rescale last PV" ,
#     12 : "accumulate sumexp" ,
#     13 : "f32tof16(P)" ,
#     14 : "tma_copy Vs" ,
#     15 : "wait Vs" ,
#     16 : "acc_o += wgmma PV" ,
#     # acc_o = acc_o / ls
#     # copy acc_o to Output
# }


op_desc = {    
    0 : "tma load Ks" ,
    1 : "wait Ks" ,
    2 : "smp = sm" ,
    3 : "clear sm" ,
    4 : "clear acc_s" ,
    5 : "acc_s = wgmma_QK" ,
    6 : "sm=reduce_max(acc_s)" ,
    7 : "sm = max(sm,smp)" ,
    8 : "name=ALU wr:_tmp_0<[2]>; / rd:smp<[2]>;" ,
    9 : "name=ALU wr:_tmp_1<[2]>; / rd:sm<[2]>;" ,
    10 : "name=ALU wr:_tmp_2<[2]>; / rd:_tmp_0<[2]>;_tmp_1<[2]>;" ,
    11 : "name=SFU wr:ss<[2]>; / rd:_tmp_2<[2]>;" ,
    12 : "name=ALU wr:_tmp_3<[32]>; / rd:acc_s<[32]>;" ,
    13 : "name=ALU wr:_tmp_4<[32]>; / rd:sm<[2]>;" ,
    14 : "name=ALU wr:_tmp_5<[32]>; / rd:_tmp_3<[32]>;_tmp_4<[32]>;" ,
    15 : "name=SFU wr:acc_s<[32]>; / rd:_tmp_5<[32]>;" ,
    16 : "name=ALU wr:ssum<[2]>; / rd:ssum<[2]>;acc_s<[32]>;" ,
    17 : "name=ALU wr:acc_o<[64]>; / rd:acc_o<[64]>;ss<[2]>;" ,
    18 : "name=ALU wr:_tmp_6<[2]>; / rd:ls<[2]>;ss<[2]>;" ,
    19 : "name=ALU wr:ls<[2]>; / rd:_tmp_6<[2]>;ssum<[2]>;" ,
    20 : "acc_s_c= cast(acc_s)" ,
    21 : "tma_load Vs" ,
    22 : "wait Vs" ,
    23 : "acc_o += wgmma_PV;" ,
}

unit_mapping = {
    'TMA': [0,21],
    'TC': [5, 23, ],
    'ALU': [1, 2, 3, 4, 6, 7,8,9, 10,  12, 13,14,16,17,18,19,20,22 ],
    'SFU': [11, 15, ]
}

BOX_HEIGHT = 0.55
LIFETIME_BOX_HEIGHT = 0.10
LIFETIME_ROW_GAP = 1.15
LIFETIME_ROW_PADDING = 0.08
LIFETIME_ITER_Y_STEP = 0.33
ISSUE_ALPHA = 0.6
UNITS_ORDER = ['TMA', 'TC', 'ALU', 'SFU']

ITERATION_COLOR_PALETTE = {
    -1: {'face': '#5c3d2e', 'edge': '#ba7a5f'},  # i-1
    0: {'face': '#1f4e5b', 'edge': '#3a889e'},   # i
    1: {'face': '#3f4f2f', 'edge': '#8aaa5e'},   # i+1
    2: {'face': '#4b3f72', 'edge': '#9d87d2'},   # i+2
    3: {'face': '#554b2f', 'edge': '#c0a45d'},   # i+3
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
# 只显示列表中的buffer lifespan
LIFETIME_BUFFER_FILTER = {'acc_s', 'acc_s_c', 'acc_o'}
LIFETIME_PREVIOUS_STAGE_CONTEXT = 1


def build_warpgroup_info(warp_assignment):
    """Return op->warpgroup and warpgroup->op descriptions."""
    op_wgid = {int(op_name[1:]): op_warp // 4 for op_name, op_warp in warp_assignment.items()}
    active_wgids = sorted(set(op_wgid.values()))
    wgid_ops = {wgid: [] for wgid in active_wgids}
    for op, wgid in sorted(op_wgid.items()):
        wgid_ops[wgid].append(op_desc[op])
    return op_wgid, active_wgids, wgid_ops


def build_y_layout():
    """Build one stable y row for each op, grouped by execution unit."""
    y_ticks_positions = []
    y_ticks_labels = []
    op_y_pos = {}
    current_y = 0
    unit_boundaries = []

    for unit in reversed(UNITS_ORDER):
        unit_ops = sorted(unit_mapping.get(unit, []))
        if not unit_ops:
            continue

        start_y_for_unit = current_y
        for op in unit_ops:
            op_y_pos[op] = current_y
            y_ticks_positions.append(current_y)
            y_ticks_labels.append(f"{unit}: s{op}")
            current_y += 1

        end_y_for_unit = current_y - 1
        unit_boundaries.append((unit, start_y_for_unit, end_y_for_unit))
        current_y += 0.6

    return op_y_pos, y_ticks_positions, y_ticks_labels, unit_boundaries, current_y


def derive_num_stages(schedule_l, ii):
    if ii <= 0:
        raise ValueError(f"II must be positive, got {ii}")
    return max(1, int(math.ceil(schedule_l / ii)))


def display_iteration_count(schedule_l, ii):
    return max(3, derive_num_stages(schedule_l, ii))


def iteration_stage_label(iter_offset):
    if iter_offset == 0:
        return 'i'
    if iter_offset > 0:
        return f"i+{iter_offset}"
    return f"i{iter_offset}"


def iteration_colors(iter_offset):
    fallback_colors = [
        {'face': '#4f3150', 'edge': '#b36cae'},
        {'face': '#2f4f4b', 'edge': '#69a89e'},
        {'face': '#50472f', 'edge': '#baa15d'},
    ]
    return ITERATION_COLOR_PALETTE.get(
        iter_offset,
        fallback_colors[abs(iter_offset) % len(fallback_colors)],
    )


def make_iteration_configs(ii, schedule_l):
    display_count = display_iteration_count(schedule_l, ii)
    return [
        {
            'start': idx * ii,
            'end': (idx + 1) * ii,
            'iter_offset': idx - 1,
        }
        for idx in range(display_count)
    ]


def build_overlap_items(schedule_m, ii, schedule_l, iteration_configs):
    plot_items = []
    stage_boundaries = {iteration['start'] for iteration in iteration_configs}
    stage_boundaries.add(iteration_configs[-1]['end'])
    focus_start = iteration_configs[-1]['start']
    all_events = set(stage_boundaries)
    all_events.update({focus_start + schedule_l})

    # 这里画的是 logical iteration 的绝对 M 时间线，不能用 M % II。
    # M % II 只适合画资源槽位表；用于依赖时间线会把跨 II 的操作折回前面。
    for iteration in iteration_configs:
        for op, start_time in schedule_m.items():
            t_start = start_time + iteration['start']
            t_end = t_start + latencies[op]
            t_issue_end = t_start + duration[op]
            plot_items.append((op, t_start, t_end, t_issue_end, iteration['iter_offset']))
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


def draw_schedule_block(ax, op, cx_start, cx_end, cx_issue_end, y_pos, colors, label, hatch=None):
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
    op_y_pos, y_ticks_positions, y_ticks_labels, unit_boundaries, total_rows = build_y_layout()
    num_stages = derive_num_stages(schedule_l, ii)
    iteration_configs = make_iteration_configs(ii, schedule_l)
    plot_items, sorted_events, time_to_coord = build_overlap_items(
        schedule_m, ii, schedule_l, iteration_configs
    )

    fig, ax = plt.subplots(figsize=(24, 10), dpi=150)
    setup_dark_axis(fig, ax)

    draw_hatches = op_wgid is not None
    for op, start, end, issue_end, iter_offset in plot_items:
        colors = iteration_colors(iter_offset)
        hatch = WG_HATCHES.get(op_wgid[op], '---') if draw_hatches else None
        label = f"{op_desc[op]} [{iteration_stage_label(iter_offset)}]"
        draw_schedule_block(
            ax,
            op,
            time_to_coord[start],
            time_to_coord[end],
            time_to_coord[issue_end],
            op_y_pos[op],
            colors,
            label,
            hatch=hatch,
        )

    draw_unit_guides(ax, len(sorted_events), unit_boundaries, y_ticks_positions, y_ticks_labels)

    important_times = {iteration['start'] for iteration in iteration_configs}
    important_times.add(iteration_configs[-1]['end'])
    important_times.add(iteration_configs[-1]['start'] + schedule_l)
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

    focus_iteration = iteration_configs[-1]
    focus_label = iteration_stage_label(focus_iteration['iter_offset'])
    axis_iteration_start_idx = time_to_coord[focus_iteration['start']]
    axis_iteration_end_idx = time_to_coord[focus_iteration['end']]
    axis_edge_color = iteration_colors(focus_iteration['iter_offset'])['edge']
    range_arrow_y = bottom_y - 1.0
    ax.annotate('', xy=(axis_iteration_start_idx, range_arrow_y), xytext=(axis_iteration_end_idx, range_arrow_y), arrowprops=dict(arrowstyle="<->", color=axis_edge_color, lw=2))
    ax.text((axis_iteration_start_idx + axis_iteration_end_idx) / 2, bottom_y - 0.9, f"Iter {focus_label}: II = {ii}", color=axis_edge_color, ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

    latency_idx = time_to_coord[focus_iteration['start'] + schedule_l]
    latency_arrow_y = bottom_y - 2.1
    ax.annotate('', xy=(axis_iteration_start_idx, latency_arrow_y), xytext=(latency_idx, latency_arrow_y), arrowprops=dict(arrowstyle="<->", color='#aaaaaa', lw=2))
    ax.text((axis_iteration_start_idx + latency_idx) / 2, bottom_y - 2.0, f"Iter {focus_label}: L = {schedule_l}, stages = ceil(L / II) = {num_stages}", color='#aaaaaa', ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

    ax.annotate('', xy=(xmax + 1.0, bottom_y), xytext=(xmin, bottom_y), arrowprops=dict(arrowstyle="->", color='white', lw=1.5))
    ax.text(xmax + 1.0, bottom_y + 0.18, 'Timeline (Hardware Cycle)', color='white', ha='right', va='bottom', fontsize=10)

    legend_patches = [
        patches.Patch(
            facecolor=iteration_colors(iteration['iter_offset'])['face'],
            edgecolor=iteration_colors(iteration['iter_offset'])['edge'],
            label=f"Iteration {iteration_stage_label(iteration['iter_offset'])} Blocks",
        )
        for iteration in iteration_configs
    ]
    legend_patches.append(
        patches.Patch(facecolor='#666666', edgecolor='none', alpha=ISSUE_ALPHA, label='Dark segment = issue duration (d); light remainder = execution')
    )
    iteration_legend = ax.legend(handles=legend_patches, loc='upper right', facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)
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
        ax.legend(handles=wg_legend_patches, loc='upper right', bbox_to_anchor=(1.0, 0.82), facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"模调度 overlap 图已成功导出: {output_filename}")


def iter_label(distance):
    if distance == 0:
        return 'i'
    return f"i-{distance}"


def draw_single_modulo_window(schedule_m, ii, schedule_l, output_filename, op_wgid=None, active_wgids=None):
    op_y_pos, y_ticks_positions, y_ticks_labels, unit_boundaries, total_rows = build_y_layout()
    single_items = []
    single_events = {0, ii, schedule_l}
    for op, start_time in schedule_m.items():
        iter_distance = start_time // ii
        slot_start = start_time % ii
        slot_issue_end = slot_start + duration[op]
        slot_end = slot_start + latencies[op]
        single_items.append((op, slot_start, slot_end, slot_issue_end, iter_distance))
        single_events.update((slot_start, slot_issue_end, slot_end))

    single_items.sort(key=lambda item: (item[1], item[4], item[0]))
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

    fig_single, ax_single = plt.subplots(figsize=(24, 10), dpi=150)
    setup_dark_axis(fig_single, ax_single)

    draw_hatches = op_wgid is not None
    for op, slot_start, slot_end, slot_issue_end, iter_distance in single_items:
        colors = single_color_palette.get(
            iter_distance,
            fallback_colors[iter_distance % len(fallback_colors)],
        )
        hatch = WG_HATCHES.get(op_wgid[op], '---') if draw_hatches else None
        label = f"{op_desc[op]} [{iter_label(iter_distance)}]"
        draw_schedule_block(
            ax_single,
            op,
            single_time_to_coord[slot_start],
            single_time_to_coord[slot_end],
            single_time_to_coord[slot_issue_end],
            op_y_pos[op],
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
        patches.Patch(facecolor='#666666', edgecolor='none', alpha=ISSUE_ALPHA, label='Dark segment = issue duration (d); light remainder = execution')
    )
    iter_legend = ax_single.legend(handles=single_iteration_legend, loc='upper right', facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)
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
        ax_single.legend(handles=single_wg_legend, loc='upper right', bbox_to_anchor=(1.0, 0.82), facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)

    plt.tight_layout()
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
            copy_live_end_exclusive = copy.get('live_end_exclusive', copy['live_end'])
            copy_live_end = copy['live_end']
            if copy_live_end is None:
                copy_live_end = copy_live_end_exclusive
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
                    'live_end': copy_live_end,
                    'live_end_exclusive': copy_live_end_exclusive,
                    'consumers': [],
                },
            )
            merged['live_start'] = min(merged['live_start'], copy['live_start'])
            merged['live_end'] = max(merged['live_end'], copy_live_end)
            merged['live_end_exclusive'] = max(merged['live_end_exclusive'], copy_live_end_exclusive)
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


def visible_lifetime_iter_offsets(schedule_l, ii):
    num_stages = derive_num_stages(schedule_l, ii)
    previous_count = min(LIFETIME_PREVIOUS_STAGE_CONTEXT, num_stages - 1)
    first_offset = -previous_count
    last_offset = num_stages - previous_count - 1
    return set(range(first_offset, last_offset + 1))


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


def build_lifetime_iter_y_positions(lifetime_items, y_by_name):
    """Pack all iteration copies of each buffer inside that buffer's row band."""
    iter_offsets_by_buffer = {}
    for item in lifetime_items:
        iter_offsets_by_buffer.setdefault(item['buffer'], set()).add(item['iter_offset'])

    y_positions = {}
    half_row = LIFETIME_ROW_GAP / 2
    slot_margin = LIFETIME_ROW_PADDING + LIFETIME_BOX_HEIGHT / 2
    for buffer_name, iter_offsets in iter_offsets_by_buffer.items():
        offsets = sorted(iter_offsets)
        row_center = y_by_name[buffer_name]
        if len(offsets) == 1:
            y_positions[(buffer_name, offsets[0])] = row_center
            continue

        y_min = row_center - half_row + slot_margin
        y_max = row_center + half_row - slot_margin
        step = (y_max - y_min) / (len(offsets) - 1)
        for idx, iter_offset in enumerate(offsets):
            y_positions[(buffer_name, iter_offset)] = y_min + idx * step

    return y_positions


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
    if LIFETIME_BUFFER_FILTER:
        lifetime_items = [
            item for item in lifetime_items
            if item['buffer'] in LIFETIME_BUFFER_FILTER
        ]
    visible_iter_offsets = visible_lifetime_iter_offsets(schedule_l, ii)
    lifetime_items = [
        item for item in lifetime_items
        if item['iter_offset'] in visible_iter_offsets
    ]
    y_by_name, y_ticks, y_labels, total_height = build_lifetime_layout(lifetime_items)
    lifetime_y_positions = build_lifetime_iter_y_positions(lifetime_items, y_by_name)
    sorted_events, time_to_coord = build_lifetime_events(lifetime_items, ii, schedule_l)

    fig, ax = plt.subplots(figsize=(26, max(10, total_height * 0.52)), dpi=150)
    setup_dark_axis(fig, ax)

    for idx in range(len(sorted_events)):
        ax.axvline(x=idx, color='#2c2c2c', linestyle=':', linewidth=0.9, zorder=1)

    buffer_row_positions = sorted(y_by_name.values())
    for lower_y, upper_y in zip(buffer_row_positions, buffer_row_positions[1:]):
        sep_y = (lower_y + upper_y) / 2
        ax.axhline(y=sep_y, color='#555555', linestyle='--', linewidth=1.1, alpha=0.75, zorder=1.5)

    for iter_offset in sorted(visible_iter_offsets):
        t = iter_offset * ii
        if t not in time_to_coord:
            continue
        label = f"{lifetime_stage_label(iter_offset)} start"
        color = lifetime_iter_colors(iter_offset)['edge']
        x = time_to_coord[t]
        ax.axvline(x=x, color=color, linestyle='-', linewidth=1.6, alpha=0.9, zorder=2)
        ax.text(x + 0.15, total_height - 0.2, label, color=color, ha='left', va='top', fontsize=9)

    for item in sorted(lifetime_items, key=lambda x: (x['buffer'], x['iter_offset'], x['live_start'])):
        y_pos = lifetime_y_positions[(item['buffer'], item['iter_offset'])]
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
    op_wgid, active_wgids, wgid_ops = build_warpgroup_info(warp_assign)
    print(wgid_ops)

    draw_overlap_schedule(
        optimized_M,
        base_I,
        optimized_L,
        'modulo_scheduling_same_line_with_desc.png',
        op_wgid=op_wgid,
        active_wgids=active_wgids,
    )
    draw_single_modulo_window(
        optimized_M,
        base_I,
        optimized_L,
        'single_iter_ops.png',
        op_wgid=op_wgid,
        active_wgids=active_wgids,
    )

    # 基础/naive 模调度方案：复用优化版 overlap 绘图格式，但不画 warp 分配 hatch。
    draw_overlap_schedule(M, I, L, 'naive_sched_plan.png')
    draw_variable_lifetimes(variable_lifetimes, base_I, optimized_L, 'var_life.png')


if __name__ == '__main__':
    main()
