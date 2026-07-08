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

warp_assign={'s0': 0, 's2': 6, 's3': 8, 's4': 7, 's5': 8, 's6': 7, 's7': 10, 's8': 6, 's9': 5, 's10': 11, 's11': 7, 's12': 4, 's13': 8, 's14': 3, 's16': 8}
optimized_L=726
base_I=256
optimized_M={0: 0, 2: 0, 3: 6, 4: 2, 5: 282, 6: 370, 7: 374, 8: 380, 9: 380, 10: 654, 11: 414, 12: 658, 13: 654, 14: 0, 16: 662}
variable_lifetimes={'Ks': {'name': 'Ks', 'storage': 'SMEM', 'buffer': 'Ks', 'footprint_bytes': 32768, 'lifetime': 'dead_on_entry', 'producers': ['s0'], 'copies': [{'producer': 's0', 'producer_warp': 0, 'iter_offset': -2, 'live_start': -512, 'live_end': -231, 'live_end_exclusive': -230, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': -230}]}, {'producer': 's0', 'producer_warp': 0, 'iter_offset': -1, 'live_start': -256, 'live_end': 25, 'live_end_exclusive': 26, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 26}]}, {'producer': 's0', 'producer_warp': 0, 'iter_offset': 0, 'live_start': 0, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 282}]}, {'producer': 's0', 'producer_warp': 0, 'iter_offset': 1, 'live_start': 256, 'live_end': 537, 'live_end_exclusive': 538, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 538}]}, {'producer': 's0', 'producer_warp': 0, 'iter_offset': 2, 'live_start': 512, 'live_end': 793, 'live_end_exclusive': 794, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 794}]}]}, 'smp': {'name': 'smp', 'storage': 'RMEM', 'buffer': 'smp', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s2'], 'copies': [{'producer': 's2', 'producer_warp': 6, 'iter_offset': -2, 'live_start': -512, 'live_end': -133, 'live_end_exclusive': -132, 'consumers': [{'consumer': 's3', 'distance': 0, 'consume_time': -506}, {'consumer': 's7', 'distance': 0, 'consume_time': -138}, {'consumer': 's8', 'distance': 0, 'consume_time': -132}]}, {'producer': 's2', 'producer_warp': 6, 'iter_offset': -1, 'live_start': -256, 'live_end': 123, 'live_end_exclusive': 124, 'consumers': [{'consumer': 's3', 'distance': 0, 'consume_time': -250}, {'consumer': 's7', 'distance': 0, 'consume_time': 118}, {'consumer': 's8', 'distance': 0, 'consume_time': 124}]}, {'producer': 's2', 'producer_warp': 6, 'iter_offset': 0, 'live_start': 0, 'live_end': 379, 'live_end_exclusive': 380, 'consumers': [{'consumer': 's3', 'distance': 0, 'consume_time': 6}, {'consumer': 's7', 'distance': 0, 'consume_time': 374}, {'consumer': 's8', 'distance': 0, 'consume_time': 380}]}, {'producer': 's2', 'producer_warp': 6, 'iter_offset': 1, 'live_start': 256, 'live_end': 635, 'live_end_exclusive': 636, 'consumers': [{'consumer': 's3', 'distance': 0, 'consume_time': 262}, {'consumer': 's7', 'distance': 0, 'consume_time': 630}, {'consumer': 's8', 'distance': 0, 'consume_time': 636}]}, {'producer': 's2', 'producer_warp': 6, 'iter_offset': 2, 'live_start': 512, 'live_end': 891, 'live_end_exclusive': 892, 'consumers': [{'consumer': 's3', 'distance': 0, 'consume_time': 518}, {'consumer': 's7', 'distance': 0, 'consume_time': 886}, {'consumer': 's8', 'distance': 0, 'consume_time': 892}]}]}, 'sm': {'name': 'sm', 'storage': 'RMEM', 'buffer': 'sm', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s3', 's6', 's7'], 'copies': [{'producer': 's3', 'producer_warp': 8, 'iter_offset': -2, 'live_start': -506, 'live_end': -143, 'live_end_exclusive': -142, 'consumers': [{'consumer': 's6', 'distance': 0, 'consume_time': -142}]}, {'producer': 's3', 'producer_warp': 8, 'iter_offset': -1, 'live_start': -250, 'live_end': 113, 'live_end_exclusive': 114, 'consumers': [{'consumer': 's6', 'distance': 0, 'consume_time': 114}]}, {'producer': 's3', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 6, 'live_end': 369, 'live_end_exclusive': 370, 'consumers': [{'consumer': 's6', 'distance': 0, 'consume_time': 370}]}, {'producer': 's3', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 262, 'live_end': 625, 'live_end_exclusive': 626, 'consumers': [{'consumer': 's6', 'distance': 0, 'consume_time': 626}]}, {'producer': 's3', 'producer_warp': 8, 'iter_offset': 2, 'live_start': 518, 'live_end': 881, 'live_end_exclusive': 882, 'consumers': [{'consumer': 's6', 'distance': 0, 'consume_time': 882}]}, {'producer': 's6', 'producer_warp': 7, 'iter_offset': -2, 'live_start': -142, 'live_end': 113, 'live_end_exclusive': 114, 'consumers': [{'consumer': 's6', 'distance': 1, 'consume_time': 114}, {'consumer': 's7', 'distance': 0, 'consume_time': -138}, {'consumer': 's9', 'distance': 0, 'consume_time': -132}]}, {'producer': 's6', 'producer_warp': 7, 'iter_offset': -1, 'live_start': 114, 'live_end': 369, 'live_end_exclusive': 370, 'consumers': [{'consumer': 's6', 'distance': 1, 'consume_time': 370}, {'consumer': 's7', 'distance': 0, 'consume_time': 118}, {'consumer': 's9', 'distance': 0, 'consume_time': 124}]}, {'producer': 's6', 'producer_warp': 7, 'iter_offset': 0, 'live_start': 370, 'live_end': 625, 'live_end_exclusive': 626, 'consumers': [{'consumer': 's6', 'distance': 1, 'consume_time': 626}, {'consumer': 's7', 'distance': 0, 'consume_time': 374}, {'consumer': 's9', 'distance': 0, 'consume_time': 380}]}, {'producer': 's6', 'producer_warp': 7, 'iter_offset': 1, 'live_start': 626, 'live_end': 881, 'live_end_exclusive': 882, 'consumers': [{'consumer': 's6', 'distance': 1, 'consume_time': 882}, {'consumer': 's7', 'distance': 0, 'consume_time': 630}, {'consumer': 's9', 'distance': 0, 'consume_time': 636}]}, {'producer': 's6', 'producer_warp': 7, 'iter_offset': 2, 'live_start': 882, 'live_end': 1137, 'live_end_exclusive': 1138, 'consumers': [{'consumer': 's6', 'distance': 1, 'consume_time': 1138}, {'consumer': 's7', 'distance': 0, 'consume_time': 886}, {'consumer': 's9', 'distance': 0, 'consume_time': 892}]}, {'producer': 's7', 'producer_warp': 10, 'iter_offset': -2, 'live_start': -138, 'live_end': 117, 'live_end_exclusive': 118, 'consumers': [{'consumer': 's7', 'distance': 1, 'consume_time': 118}, {'consumer': 's8', 'distance': 0, 'consume_time': -132}, {'consumer': 's9', 'distance': 0, 'consume_time': -132}]}, {'producer': 's7', 'producer_warp': 10, 'iter_offset': -1, 'live_start': 118, 'live_end': 373, 'live_end_exclusive': 374, 'consumers': [{'consumer': 's7', 'distance': 1, 'consume_time': 374}, {'consumer': 's8', 'distance': 0, 'consume_time': 124}, {'consumer': 's9', 'distance': 0, 'consume_time': 124}]}, {'producer': 's7', 'producer_warp': 10, 'iter_offset': 0, 'live_start': 374, 'live_end': 629, 'live_end_exclusive': 630, 'consumers': [{'consumer': 's7', 'distance': 1, 'consume_time': 630}, {'consumer': 's8', 'distance': 0, 'consume_time': 380}, {'consumer': 's9', 'distance': 0, 'consume_time': 380}]}, {'producer': 's7', 'producer_warp': 10, 'iter_offset': 1, 'live_start': 630, 'live_end': 885, 'live_end_exclusive': 886, 'consumers': [{'consumer': 's7', 'distance': 1, 'consume_time': 886}, {'consumer': 's8', 'distance': 0, 'consume_time': 636}, {'consumer': 's9', 'distance': 0, 'consume_time': 636}]}, {'producer': 's7', 'producer_warp': 10, 'iter_offset': 2, 'live_start': 886, 'live_end': 1141, 'live_end_exclusive': 1142, 'consumers': [{'consumer': 's7', 'distance': 1, 'consume_time': 1142}, {'consumer': 's8', 'distance': 0, 'consume_time': 892}, {'consumer': 's9', 'distance': 0, 'consume_time': 892}]}]}, 'acc_s': {'name': 'acc_s', 'storage': 'RMEM', 'buffer': 'acc_s', 'footprint_bytes': 128, 'lifetime': 'dead_on_entry', 'producers': ['s4', 's5', 's9'], 'copies': [{'producer': 's4', 'producer_warp': 7, 'iter_offset': -2, 'live_start': -510, 'live_end': -231, 'live_end_exclusive': -230, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': -230}]}, {'producer': 's4', 'producer_warp': 7, 'iter_offset': -1, 'live_start': -254, 'live_end': 25, 'live_end_exclusive': 26, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 26}]}, {'producer': 's4', 'producer_warp': 7, 'iter_offset': 0, 'live_start': 2, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 282}]}, {'producer': 's4', 'producer_warp': 7, 'iter_offset': 1, 'live_start': 258, 'live_end': 537, 'live_end_exclusive': 538, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 538}]}, {'producer': 's4', 'producer_warp': 7, 'iter_offset': 2, 'live_start': 514, 'live_end': 793, 'live_end_exclusive': 794, 'consumers': [{'consumer': 's5', 'distance': 0, 'consume_time': 794}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': -2, 'live_start': -230, 'live_end': 25, 'live_end_exclusive': 26, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 26}, {'consumer': 's6', 'distance': 0, 'consume_time': -142}, {'consumer': 's9', 'distance': 0, 'consume_time': -132}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 26, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 282}, {'consumer': 's6', 'distance': 0, 'consume_time': 114}, {'consumer': 's9', 'distance': 0, 'consume_time': 124}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 282, 'live_end': 537, 'live_end_exclusive': 538, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 538}, {'consumer': 's6', 'distance': 0, 'consume_time': 370}, {'consumer': 's9', 'distance': 0, 'consume_time': 380}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 538, 'live_end': 793, 'live_end_exclusive': 794, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 794}, {'consumer': 's6', 'distance': 0, 'consume_time': 626}, {'consumer': 's9', 'distance': 0, 'consume_time': 636}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 2, 'live_start': 794, 'live_end': 1049, 'live_end_exclusive': 1050, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 1050}, {'consumer': 's6', 'distance': 0, 'consume_time': 882}, {'consumer': 's9', 'distance': 0, 'consume_time': 892}]}, {'producer': 's9', 'producer_warp': 5, 'iter_offset': -2, 'live_start': -132, 'live_end': 141, 'live_end_exclusive': 142, 'consumers': [{'consumer': 's9', 'distance': 1, 'consume_time': 124}, {'consumer': 's10', 'distance': 0, 'consume_time': 142}, {'consumer': 's13', 'distance': 0, 'consume_time': 142}]}, {'producer': 's9', 'producer_warp': 5, 'iter_offset': -1, 'live_start': 124, 'live_end': 397, 'live_end_exclusive': 398, 'consumers': [{'consumer': 's9', 'distance': 1, 'consume_time': 380}, {'consumer': 's10', 'distance': 0, 'consume_time': 398}, {'consumer': 's13', 'distance': 0, 'consume_time': 398}]}, {'producer': 's9', 'producer_warp': 5, 'iter_offset': 0, 'live_start': 380, 'live_end': 653, 'live_end_exclusive': 654, 'consumers': [{'consumer': 's9', 'distance': 1, 'consume_time': 636}, {'consumer': 's10', 'distance': 0, 'consume_time': 654}, {'consumer': 's13', 'distance': 0, 'consume_time': 654}]}, {'producer': 's9', 'producer_warp': 5, 'iter_offset': 1, 'live_start': 636, 'live_end': 909, 'live_end_exclusive': 910, 'consumers': [{'consumer': 's9', 'distance': 1, 'consume_time': 892}, {'consumer': 's10', 'distance': 0, 'consume_time': 910}, {'consumer': 's13', 'distance': 0, 'consume_time': 910}]}, {'producer': 's9', 'producer_warp': 5, 'iter_offset': 2, 'live_start': 892, 'live_end': 1165, 'live_end_exclusive': 1166, 'consumers': [{'consumer': 's9', 'distance': 1, 'consume_time': 1148}, {'consumer': 's10', 'distance': 0, 'consume_time': 1166}, {'consumer': 's13', 'distance': 0, 'consume_time': 1166}]}]}, 'desc_a': {'name': 'desc_a', 'storage': 'RMEM', 'buffer': 'desc_a', 'footprint_bytes': 2, 'lifetime': 'dead_on_entry', 'producers': ['s5', 's16'], 'copies': [{'producer': 's5', 'producer_warp': 8, 'iter_offset': -2, 'live_start': -230, 'live_end': 25, 'live_end_exclusive': 26, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 26}, {'consumer': 's6', 'distance': 0, 'consume_time': -142}, {'consumer': 's9', 'distance': 0, 'consume_time': -132}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 26, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 282}, {'consumer': 's6', 'distance': 0, 'consume_time': 114}, {'consumer': 's9', 'distance': 0, 'consume_time': 124}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 282, 'live_end': 537, 'live_end_exclusive': 538, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 538}, {'consumer': 's6', 'distance': 0, 'consume_time': 370}, {'consumer': 's9', 'distance': 0, 'consume_time': 380}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 538, 'live_end': 793, 'live_end_exclusive': 794, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 794}, {'consumer': 's6', 'distance': 0, 'consume_time': 626}, {'consumer': 's9', 'distance': 0, 'consume_time': 636}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 2, 'live_start': 794, 'live_end': 1049, 'live_end_exclusive': 1050, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 1050}, {'consumer': 's6', 'distance': 0, 'consume_time': 882}, {'consumer': 's9', 'distance': 0, 'consume_time': 892}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': -2, 'live_start': 150, 'live_end': 405, 'live_end_exclusive': 406, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 406}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 406, 'live_end': 661, 'live_end_exclusive': 662, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 662}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 662, 'live_end': 917, 'live_end_exclusive': 918, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 918}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 918, 'live_end': 1173, 'live_end_exclusive': 1174, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 1174}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 2, 'live_start': 1174, 'live_end': 1429, 'live_end_exclusive': 1430, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 1430}]}]}, 'desc_b': {'name': 'desc_b', 'storage': 'RMEM', 'buffer': 'desc_b', 'footprint_bytes': 2, 'lifetime': 'dead_on_entry', 'producers': ['s5', 's16'], 'copies': [{'producer': 's5', 'producer_warp': 8, 'iter_offset': -2, 'live_start': -230, 'live_end': 25, 'live_end_exclusive': 26, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 26}, {'consumer': 's6', 'distance': 0, 'consume_time': -142}, {'consumer': 's9', 'distance': 0, 'consume_time': -132}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 26, 'live_end': 281, 'live_end_exclusive': 282, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 282}, {'consumer': 's6', 'distance': 0, 'consume_time': 114}, {'consumer': 's9', 'distance': 0, 'consume_time': 124}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 282, 'live_end': 537, 'live_end_exclusive': 538, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 538}, {'consumer': 's6', 'distance': 0, 'consume_time': 370}, {'consumer': 's9', 'distance': 0, 'consume_time': 380}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 538, 'live_end': 793, 'live_end_exclusive': 794, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 794}, {'consumer': 's6', 'distance': 0, 'consume_time': 626}, {'consumer': 's9', 'distance': 0, 'consume_time': 636}]}, {'producer': 's5', 'producer_warp': 8, 'iter_offset': 2, 'live_start': 794, 'live_end': 1049, 'live_end_exclusive': 1050, 'consumers': [{'consumer': 's5', 'distance': 1, 'consume_time': 1050}, {'consumer': 's6', 'distance': 0, 'consume_time': 882}, {'consumer': 's9', 'distance': 0, 'consume_time': 892}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': -2, 'live_start': 150, 'live_end': 405, 'live_end_exclusive': 406, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 406}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 406, 'live_end': 661, 'live_end_exclusive': 662, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 662}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 662, 'live_end': 917, 'live_end_exclusive': 918, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 918}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 918, 'live_end': 1173, 'live_end_exclusive': 1174, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 1174}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 2, 'live_start': 1174, 'live_end': 1429, 'live_end_exclusive': 1430, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 1430}]}]}, 'ss': {'name': 'ss', 'storage': 'RMEM', 'buffer': 'ss', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s8'], 'copies': [{'producer': 's8', 'producer_warp': 6, 'iter_offset': -2, 'live_start': -132, 'live_end': 145, 'live_end_exclusive': 146, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 124}, {'consumer': 's11', 'distance': 0, 'consume_time': -98}, {'consumer': 's12', 'distance': 0, 'consume_time': 146}]}, {'producer': 's8', 'producer_warp': 6, 'iter_offset': -1, 'live_start': 124, 'live_end': 401, 'live_end_exclusive': 402, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 380}, {'consumer': 's11', 'distance': 0, 'consume_time': 158}, {'consumer': 's12', 'distance': 0, 'consume_time': 402}]}, {'producer': 's8', 'producer_warp': 6, 'iter_offset': 0, 'live_start': 380, 'live_end': 657, 'live_end_exclusive': 658, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 636}, {'consumer': 's11', 'distance': 0, 'consume_time': 414}, {'consumer': 's12', 'distance': 0, 'consume_time': 658}]}, {'producer': 's8', 'producer_warp': 6, 'iter_offset': 1, 'live_start': 636, 'live_end': 913, 'live_end_exclusive': 914, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 892}, {'consumer': 's11', 'distance': 0, 'consume_time': 670}, {'consumer': 's12', 'distance': 0, 'consume_time': 914}]}, {'producer': 's8', 'producer_warp': 6, 'iter_offset': 2, 'live_start': 892, 'live_end': 1169, 'live_end_exclusive': 1170, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 1148}, {'consumer': 's11', 'distance': 0, 'consume_time': 926}, {'consumer': 's12', 'distance': 0, 'consume_time': 1170}]}]}, 'ss_shared': {'name': 'ss_shared', 'storage': 'SMEM', 'buffer': 'ss_shared', 'footprint_bytes': 512, 'lifetime': 'dead_on_entry', 'producers': ['s8'], 'copies': [{'producer': 's8', 'producer_warp': 6, 'iter_offset': -2, 'live_start': -132, 'live_end': 145, 'live_end_exclusive': 146, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 124}, {'consumer': 's11', 'distance': 0, 'consume_time': -98}, {'consumer': 's12', 'distance': 0, 'consume_time': 146}]}, {'producer': 's8', 'producer_warp': 6, 'iter_offset': -1, 'live_start': 124, 'live_end': 401, 'live_end_exclusive': 402, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 380}, {'consumer': 's11', 'distance': 0, 'consume_time': 158}, {'consumer': 's12', 'distance': 0, 'consume_time': 402}]}, {'producer': 's8', 'producer_warp': 6, 'iter_offset': 0, 'live_start': 380, 'live_end': 657, 'live_end_exclusive': 658, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 636}, {'consumer': 's11', 'distance': 0, 'consume_time': 414}, {'consumer': 's12', 'distance': 0, 'consume_time': 658}]}, {'producer': 's8', 'producer_warp': 6, 'iter_offset': 1, 'live_start': 636, 'live_end': 913, 'live_end_exclusive': 914, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 892}, {'consumer': 's11', 'distance': 0, 'consume_time': 670}, {'consumer': 's12', 'distance': 0, 'consume_time': 914}]}, {'producer': 's8', 'producer_warp': 6, 'iter_offset': 2, 'live_start': 892, 'live_end': 1169, 'live_end_exclusive': 1170, 'consumers': [{'consumer': 's8', 'distance': 1, 'consume_time': 1148}, {'consumer': 's11', 'distance': 0, 'consume_time': 926}, {'consumer': 's12', 'distance': 0, 'consume_time': 1170}]}]}, 'ssum': {'name': 'ssum', 'storage': 'RMEM', 'buffer': 'ssum', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s10'], 'copies': [{'producer': 's10', 'producer_warp': 11, 'iter_offset': -2, 'live_start': 142, 'live_end': 397, 'live_end_exclusive': 398, 'consumers': [{'consumer': 's10', 'distance': 1, 'consume_time': 398}, {'consumer': 's12', 'distance': 0, 'consume_time': 146}]}, {'producer': 's10', 'producer_warp': 11, 'iter_offset': -1, 'live_start': 398, 'live_end': 653, 'live_end_exclusive': 654, 'consumers': [{'consumer': 's10', 'distance': 1, 'consume_time': 654}, {'consumer': 's12', 'distance': 0, 'consume_time': 402}]}, {'producer': 's10', 'producer_warp': 11, 'iter_offset': 0, 'live_start': 654, 'live_end': 909, 'live_end_exclusive': 910, 'consumers': [{'consumer': 's10', 'distance': 1, 'consume_time': 910}, {'consumer': 's12', 'distance': 0, 'consume_time': 658}]}, {'producer': 's10', 'producer_warp': 11, 'iter_offset': 1, 'live_start': 910, 'live_end': 1165, 'live_end_exclusive': 1166, 'consumers': [{'consumer': 's10', 'distance': 1, 'consume_time': 1166}, {'consumer': 's12', 'distance': 0, 'consume_time': 914}]}, {'producer': 's10', 'producer_warp': 11, 'iter_offset': 2, 'live_start': 1166, 'live_end': 1421, 'live_end_exclusive': 1422, 'consumers': [{'consumer': 's10', 'distance': 1, 'consume_time': 1422}, {'consumer': 's12', 'distance': 0, 'consume_time': 1170}]}]}, 'acc_o': {'name': 'acc_o', 'storage': 'RMEM', 'buffer': 'acc_o', 'footprint_bytes': 256, 'lifetime': 'dead_on_entry', 'producers': ['s11', 's16'], 'copies': [{'producer': 's11', 'producer_warp': 7, 'iter_offset': -2, 'live_start': -98, 'live_end': 157, 'live_end_exclusive': 158, 'consumers': [{'consumer': 's11', 'distance': 1, 'consume_time': 158}, {'consumer': 's16', 'distance': 0, 'consume_time': 150}]}, {'producer': 's11', 'producer_warp': 7, 'iter_offset': -1, 'live_start': 158, 'live_end': 413, 'live_end_exclusive': 414, 'consumers': [{'consumer': 's11', 'distance': 1, 'consume_time': 414}, {'consumer': 's16', 'distance': 0, 'consume_time': 406}]}, {'producer': 's11', 'producer_warp': 7, 'iter_offset': 0, 'live_start': 414, 'live_end': 669, 'live_end_exclusive': 670, 'consumers': [{'consumer': 's11', 'distance': 1, 'consume_time': 670}, {'consumer': 's16', 'distance': 0, 'consume_time': 662}]}, {'producer': 's11', 'producer_warp': 7, 'iter_offset': 1, 'live_start': 670, 'live_end': 925, 'live_end_exclusive': 926, 'consumers': [{'consumer': 's11', 'distance': 1, 'consume_time': 926}, {'consumer': 's16', 'distance': 0, 'consume_time': 918}]}, {'producer': 's11', 'producer_warp': 7, 'iter_offset': 2, 'live_start': 926, 'live_end': 1181, 'live_end_exclusive': 1182, 'consumers': [{'consumer': 's11', 'distance': 1, 'consume_time': 1182}, {'consumer': 's16', 'distance': 0, 'consume_time': 1174}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': -2, 'live_start': 150, 'live_end': 405, 'live_end_exclusive': 406, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 406}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 406, 'live_end': 661, 'live_end_exclusive': 662, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 662}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 662, 'live_end': 917, 'live_end_exclusive': 918, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 918}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 918, 'live_end': 1173, 'live_end_exclusive': 1174, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 1174}]}, {'producer': 's16', 'producer_warp': 8, 'iter_offset': 2, 'live_start': 1174, 'live_end': 1429, 'live_end_exclusive': 1430, 'consumers': [{'consumer': 's16', 'distance': 1, 'consume_time': 1430}]}]}, 'ls': {'name': 'ls', 'storage': 'RMEM', 'buffer': 'ls', 'footprint_bytes': 8, 'lifetime': 'dead_on_entry', 'producers': ['s12'], 'copies': [{'producer': 's12', 'producer_warp': 4, 'iter_offset': -2, 'live_start': 146, 'live_end': 401, 'live_end_exclusive': 402, 'consumers': [{'consumer': 's12', 'distance': 1, 'consume_time': 402}]}, {'producer': 's12', 'producer_warp': 4, 'iter_offset': -1, 'live_start': 402, 'live_end': 657, 'live_end_exclusive': 658, 'consumers': [{'consumer': 's12', 'distance': 1, 'consume_time': 658}]}, {'producer': 's12', 'producer_warp': 4, 'iter_offset': 0, 'live_start': 658, 'live_end': 913, 'live_end_exclusive': 914, 'consumers': [{'consumer': 's12', 'distance': 1, 'consume_time': 914}]}, {'producer': 's12', 'producer_warp': 4, 'iter_offset': 1, 'live_start': 914, 'live_end': 1169, 'live_end_exclusive': 1170, 'consumers': [{'consumer': 's12', 'distance': 1, 'consume_time': 1170}]}, {'producer': 's12', 'producer_warp': 4, 'iter_offset': 2, 'live_start': 1170, 'live_end': 1425, 'live_end_exclusive': 1426, 'consumers': [{'consumer': 's12', 'distance': 1, 'consume_time': 1426}]}]}, 'acc_s_c': {'name': 'acc_s_c', 'storage': 'SMEM', 'buffer': 'acc_s_c', 'footprint_bytes': 16384, 'lifetime': 'dead_on_entry', 'producers': ['s13'], 'copies': [{'producer': 's13', 'producer_warp': 8, 'iter_offset': -2, 'live_start': 142, 'live_end': 149, 'live_end_exclusive': 150, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 150}]}, {'producer': 's13', 'producer_warp': 8, 'iter_offset': -1, 'live_start': 398, 'live_end': 405, 'live_end_exclusive': 406, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 406}]}, {'producer': 's13', 'producer_warp': 8, 'iter_offset': 0, 'live_start': 654, 'live_end': 661, 'live_end_exclusive': 662, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 662}]}, {'producer': 's13', 'producer_warp': 8, 'iter_offset': 1, 'live_start': 910, 'live_end': 917, 'live_end_exclusive': 918, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 918}]}, {'producer': 's13', 'producer_warp': 8, 'iter_offset': 2, 'live_start': 1166, 'live_end': 1173, 'live_end_exclusive': 1174, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 1174}]}]}, 'Vs': {'name': 'Vs', 'storage': 'SMEM', 'buffer': 'Vs', 'footprint_bytes': 32768, 'lifetime': 'dead_on_entry', 'producers': ['s14'], 'copies': [{'producer': 's14', 'producer_warp': 3, 'iter_offset': -2, 'live_start': -512, 'live_end': 149, 'live_end_exclusive': 150, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 150}]}, {'producer': 's14', 'producer_warp': 3, 'iter_offset': -1, 'live_start': -256, 'live_end': 405, 'live_end_exclusive': 406, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 406}]}, {'producer': 's14', 'producer_warp': 3, 'iter_offset': 0, 'live_start': 0, 'live_end': 661, 'live_end_exclusive': 662, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 662}]}, {'producer': 's14', 'producer_warp': 3, 'iter_offset': 1, 'live_start': 256, 'live_end': 917, 'live_end_exclusive': 918, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 918}]}, {'producer': 's14', 'producer_warp': 3, 'iter_offset': 2, 'live_start': 512, 'live_end': 1173, 'live_end_exclusive': 1174, 'consumers': [{'consumer': 's16', 'distance': 0, 'consume_time': 1174}]}]}}

# naive mod sched
I = 512
L = 1174
M = {0: 0, 2: 432, 3: 426, 4: 214, 5: 282, 6: 434, 7: 440, 8: 864, 9: 448, 10: 1164, 11: 914, 12: 1170, 13: 978, 14: 764, 16: 1046}

# ========================================
# 执行+发射延迟
latencies={0: 282, 1: 1, 2: 8, 3: 8, 4: 68, 5: 152, 6: 6, 7: 8, 8: 50, 9: 530, 10: 6, 11: 132, 12: 8, 13: 68, 14: 282, 15: 1, 16: 176}
# 发射延迟
duration={0: 2, 1: 1, 2: 4, 3: 4, 4: 64, 5: 128, 6: 4, 7: 4, 8: 32, 9: 512, 10: 4, 11: 128, 12: 4, 13: 64, 14: 2, 15: 1, 16: 128}
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

# op_desc = {
#     0 : "async_copy Ks" ,
#     1 : "wait Ks" ,
#     2 : "smp = sm" ,
#     3 : "clear sm" ,
#     4 : "clear acc_s" ,
#     5 : "acc_s = wgmmaQK" ,
#     6 : "sm = reduce_max(acc_s)" ,
#     7 : "sm = max(sm, smp)  // update global max" ,
#     8 : "ss = exp2(smp - sm) // get scale" ,
#     9 : "acc_s = exp2(acc_s - sm)  // softmax's exp" ,
#     10 : "ssum = reduce_sum(acc_s ) // softmax's sumexp" ,
#     11 : "acc_o =  acc_o * ss // rescale last PV" ,
#     12 : "ls=ls * ss + ssum // accumulate sumexp" ,
#     13 : "acc_s_c = castf32tof16(acc_s)" ,
#     14 : "async_copy Vs" ,
#     15 : "wait Vs" ,
#     16 : "acc_o += wgmma(acc_s_c, V)" ,
#     # acc_o = acc_o / ls
#     # copy acc_o to Output
# }

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
        for op, start_time in schedule_m.items():
            t_start = start_time + iteration['start']
            t_end = t_start + latencies[op]
            t_issue_end = t_start + duration[op]
            plot_items.append((op, t_start, t_end, t_issue_end, iteration['stage']))
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
    iteration_configs = make_iteration_configs(ii)
    plot_items, sorted_events, time_to_coord = build_overlap_items(
        schedule_m, ii, schedule_l, iteration_configs
    )

    fig, ax = plt.subplots(figsize=(24, 10), dpi=150)
    setup_dark_axis(fig, ax)

    draw_hatches = op_wgid is not None
    for op, start, end, issue_end, stage in plot_items:
        colors = ITERATION_COLOR_PALETTE[stage]
        hatch = WG_HATCHES.get(op_wgid[op], '---') if draw_hatches else None
        label = f"{op_desc[op]} [{STAGE_LABELS[stage]}]"
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
        patches.Patch(facecolor='#666666', edgecolor='none', alpha=ISSUE_ALPHA, label='Dark segment = issue duration (d); light remainder = execution'),
    ]
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
