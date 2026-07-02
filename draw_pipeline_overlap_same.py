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
warp_assign={'s0': 2, 's2': 6, 's3': 6, 's4': 6, 's5': 4, 's6': 5, 's7': 4, 's8': 5, 's9': 7, 's10': 6, 's11': 5, 's12': 6, 's13': 4, 's14': 2, 's16': 4}
optimized_L=1174
base_I=512
optimized_M={0: 0, 2: 8, 3: 4, 4: 12, 5: 282, 6: 434, 7: 440, 8: 448, 9: 448, 10: 978, 11: 498, 12: 984, 13: 978, 14: 2, 16: 1046}

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
ISSUE_ALPHA = 0.8
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


if __name__ == '__main__':
    main()
