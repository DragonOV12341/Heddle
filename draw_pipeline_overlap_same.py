import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ----------------- 1. 原始数据与配置 -----------------

warp_assign={'s0': 7, 's1': 2, 's2': 1, 's3': 1, 's4': 3, 's5': 0, 's6': 0, 's7': 0, 's8': 3, 's9': 2, 's10': 1, 's11': 3, 's12': 1, 's13': 0, 's14': 7, 's15': 1, 's16': 0}
optimized_L=1523
base_I=514
optimized_M={0: 0, 1: 282, 2: 0, 3: 4, 4: 4, 5: 495, 6: 647, 7: 653, 8: 661, 9: 797, 10: 1327, 11: 797, 12: 1333, 13: 1327, 14: 2, 15: 795, 16: 1395}

# ========================================
# 执行+发射延迟
latencies={0: 282, 1: 1, 2: 8, 3: 8, 4: 68, 5: 152, 6: 6, 7: 8, 8: 50, 9: 530, 10: 6, 11: 132, 12: 8, 13: 68, 14: 282, 15: 1, 16: 176}
# 发射延迟
duration={0: 2, 1: 1, 2: 4, 3: 4, 4: 64, 5: 128, 6: 4, 7: 4, 8: 32, 9: 512, 10: 4, 11: 128, 12: 4, 13: 64, 14: 2, 15: 1, 16: 128}
op_desc = {
    0 : 'async_copy Ks',
    1 : 'wait Ks',
    2 : 'smp = sm',
    3 : 'clear sm',
    4 : 'clear acc_s',
    5 : 'wgmmaQK',
    6 : 'sm = reduce_max QK',
    7 : 'sm = max(sm, smp)',
    8 : 'ss = exp2(smp, sm)',
    9 : 'acc_s = exp2(acc_s, sm)',
    10 : 'ssum = acc_s / ssum',
    11 : 'acc_o = f11(acc_o, ss)',
    12 : 'ls = f12(ls, ss, ssum)',
    13 : 'acc_s_c = cast(acc_s)',
    14 : 'async_copy Vs',
    15 : 'wait Vs',
    16 : 'acc_s_o += wgmmaPV',
}

unit_mapping = {
    'TMA': [0, 1, 14, 15],
    'TC': [5, 16, ],
    'ALU': [2, 3, 4, 6, 7, 10, 11, 12, 13, ],
    'SFU': [8, 9, ]
}

# warpgroupId - 运行的ops。warp_assign 里 sN 的 warpId 通过 warpId // 4 映射到 warpgroup。
op_wgid = {int(op_name[1:]): op_warp // 4 for op_name, op_warp in warp_assign.items()}
active_wgids = sorted(set(op_wgid.values()))
wgid_ops = {wgid: [] for wgid in active_wgids}

for op, wgid in sorted(op_wgid.items()):
    wgid_ops[wgid].append(op_desc[op])

print(wgid_ops)

# 绘图派生参数。后续标注里的 II、迭代窗口、单迭代 latency 都从这里取，
# 避免换一组调度结果时还要手动改图中文字里的数字。
prev_iteration_offset = 0
curr_iteration_offset = base_I
next_iteration_offset = base_I * 2
future_iteration_offset = base_I * 3
single_iteration_latency = optimized_L
curr_iteration_latency_end = curr_iteration_offset + single_iteration_latency
next_iteration_latency_end = next_iteration_offset + single_iteration_latency

prev_iteration_stage = 1
curr_iteration_stage = 0
next_iteration_stage = 2

stage_labels = {
    prev_iteration_stage: 'i-1',
    curr_iteration_stage: 'i',
    next_iteration_stage: 'i+1',
}

iteration_configs = [
    {
        'start': prev_iteration_offset,
        'end': curr_iteration_offset,
        'label': f"Iteration I-1 Range (II = {base_I})",
        'stage': prev_iteration_stage,
    },
    {
        'start': curr_iteration_offset,
        'end': next_iteration_offset,
        'label': f"Iteration I Range (II = {base_I})",
        'stage': curr_iteration_stage,
    },
    {
        'start': next_iteration_offset,
        'end': future_iteration_offset,
        'label': f"Iteration I+1 Range (II = {base_I})",
        'stage': next_iteration_stage,
    },
]

# ----------------- 2. 生成双迭代全操作时间节点 -----------------
plot_items = []
all_events = set()

all_events.add(prev_iteration_offset)
all_events.add(curr_iteration_offset)
all_events.add(next_iteration_offset)
all_events.add(future_iteration_offset)
all_events.add(curr_iteration_latency_end)
all_events.add(next_iteration_latency_end)

# 铺平迭代 I-1、I、I+1 的全部分量。
# 注意：这里画的是同一 logical iteration 的绝对 M 时间线，不能用 M % II。
# M % II 只适合画资源槽位表；用于依赖时间线会把跨 II 的操作折回前面，
# 造成 PV[i] 看起来和 QK[i] 错误重叠。
for iteration in iteration_configs:
    for op, start_time in optimized_M.items():
        lat = latencies[op]
        issue_duration = duration[op]
        t_start = start_time + iteration['start']
        t_end = t_start + lat
        t_issue_end = t_start + issue_duration
        plot_items.append((op, t_start, t_end, t_issue_end, iteration['stage']))
        all_events.add(t_start)
        all_events.add(t_end)
        all_events.add(t_issue_end)

sorted_events = sorted(list(all_events))
time_to_coord = {t: i for i, t in enumerate(sorted_events)}

# ----------------- 3. 构建合并 Y 轴（同一 op 共享同一行） -----------------
units_order = ['TMA', 'TC', 'ALU', 'SFU']
y_ticks_positions = []
y_ticks_labels = []
op_y_pos = {}

current_y = 0
unit_boundaries = []

for unit in reversed(units_order):
    unit_ops = sorted(unit_mapping.get(unit, []))
    if not unit_ops:
        continue
    
    start_y_for_unit = current_y
    for op in unit_ops:
        op_y_pos[op] = current_y
        y_ticks_positions.append(current_y)
        # 侧边行名保持精炼，方便矩形内承载长文本
        y_ticks_labels.append(f"{unit}: s{op}")
        current_y += 1
        
    end_y_for_unit = current_y - 1
    unit_boundaries.append((unit, start_y_for_unit, end_y_for_unit))
    current_y += 0.6

total_rows = current_y

# ----------------- 4. 绘图和色彩渲染 -----------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(24, 10), dpi=150)
ax.set_facecolor('#121212')
fig.patch.set_facecolor('#121212')

box_height = 0.55
issue_alpha = 0.8
color_palette = {
    prev_iteration_stage: {'face': '#5c3d2e', 'edge': '#ba7a5f'},  # 迭代 I-1 [i-1] -> 深棕
    curr_iteration_stage: {'face': '#1f4e5b', 'edge': '#3a889e'},  # 迭代 I [i]     -> 深青
    next_iteration_stage: {'face': '#3f4f2f', 'edge': '#8aaa5e'},  # 迭代 I+1 [i+1] -> 深绿
}
wg_hatches = {
    0: '///',
    1: '',
    2: '...',
    3: '\\\\\\',
}

for op, start, end, issue_end, stage in plot_items:
    y_pos = op_y_pos[op] # 确保同一 op 在同一行
    
    # 保留事件点压缩坐标，让短 op 仍有可读空间；坐标标签继续显示真实 hardware cycle。
    # end = start + latencies[op] 表示发射+执行，issue_end = start + duration[op] 表示发射段。
    cx_start = time_to_coord[start]
    cx_end = time_to_coord[end]
    cx_issue_end = time_to_coord[issue_end]
    c_width = cx_end - cx_start
    issue_width = cx_issue_end - cx_start
    
    colors = color_palette[stage]
    wgid = op_wgid[op]
    hatch = wg_hatches.get(wgid, '---')
    
    # 浅色整段表示总延迟 L = issue + execute；深色前段表示 issue duration。
    rect = patches.Rectangle(
        (cx_start, y_pos - box_height / 2),
        c_width, box_height, 
        linewidth=1.5, edgecolor=colors['edge'], facecolor=colors['edge'], zorder=3, alpha=0.28
    )
    ax.add_patch(rect)

    issue_rect = patches.Rectangle(
        (cx_start, y_pos - box_height / 2),
        issue_width, box_height,
        linewidth=0, edgecolor='none', facecolor=colors['face'], zorder=4, alpha=issue_alpha
    )
    ax.add_patch(issue_rect)

    # 纹理只表达 warpgroup，不改变 iteration 颜色语义。
    hatch_rect = patches.Rectangle(
        (cx_start, y_pos - box_height / 2),
        c_width, box_height,
        linewidth=0, edgecolor='#d8d8d8', facecolor='none', hatch=hatch, zorder=4.5, alpha=0.3
    )
    ax.add_patch(hatch_rect)
    
    # 【核心恢复】将完整的算子指令描述 (desc) 加上迭代标签写进滑块内
    desc = op_desc[op]
    # label = f"{desc} [{stage_labels[stage]}] d={duration[op]} L={latencies[op]}"
    label = f"{desc} [{stage_labels[stage]}]"
    ax.text(
        cx_start + c_width / 2, y_pos, label, 
        color='white', ha='center', va='center', fontsize=8, zorder=5, fontweight='bold', alpha=0.85
    )

# ----------------- 5. 辅助网格和单元边界线 -----------------
for idx in range(len(sorted_events)):
    ax.axvline(x=idx, color='#2c2c2c', linestyle=':', linewidth=1.0, zorder=1)

for i in range(len(unit_boundaries) - 1):
    end_y = unit_boundaries[i][2]
    next_start_y = unit_boundaries[i+1][1]
    sep_y = (end_y + next_start_y) / 2
    ax.axhline(y=sep_y, color='#555555', linestyle='--', linewidth=1.2, zorder=1)

ax.set_yticks(y_ticks_positions)
ax.set_yticklabels(y_ticks_labels, fontsize=9)

max_xtick_labels = 34
tick_step = max(1, int(np.ceil(len(sorted_events) / max_xtick_labels)))
important_times = {
    prev_iteration_offset,
    curr_iteration_offset,
    next_iteration_offset,
    future_iteration_offset,
    next_iteration_latency_end,
}
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
min_tick_gap = 3
xtick_positions = sorted(important_positions)
for idx in sorted(sampled_positions):
    if all(abs(idx - important_idx) >= min_tick_gap for important_idx in important_positions):
        xtick_positions.append(idx)
xtick_positions = sorted(set(xtick_positions))
ax.set_xticks(xtick_positions)
ax.set_xticklabels([str(sorted_events[idx]) for idx in xtick_positions], fontsize=8, rotation=0)
ax.tick_params(axis='x', pad=8, length=4)

# ----------------- 6. 底部时轴与架构参数标注（II 与 L） -----------------
bottom_y = -1.0
ax.spines['bottom'].set_position(('data', bottom_y))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

xmin, xmax = -0.5, len(sorted_events) - 0.5
ax.set_xlim(xmin, xmax + 1.5)
ax.set_ylim(bottom_y - 3.2, total_rows - 0.2)

axis_iteration_start_idx = time_to_coord[next_iteration_offset]

range_arrow_y = bottom_y - 1.0
range_text_y = bottom_y - 0.9
axis_iteration_item = next(item for item in iteration_configs if item['stage'] == next_iteration_stage)
axis_iteration_end_idx = time_to_coord[axis_iteration_item['end']]
axis_edge_color = color_palette[next_iteration_stage]['edge']
ax.annotate('', xy=(axis_iteration_start_idx, range_arrow_y), xytext=(axis_iteration_end_idx, range_arrow_y), arrowprops=dict(arrowstyle="<->", color=axis_edge_color, lw=2))
ax.text((axis_iteration_start_idx + axis_iteration_end_idx) / 2, range_text_y, f"Iter I+1: II = {base_I}", color=axis_edge_color, ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

# C. 标注单个核心循环的完整处理生命周期 Latency L
latency_idx = time_to_coord[next_iteration_latency_end]
latency_arrow_y = bottom_y - 2.1
latency_text_y = bottom_y - 2.0
ax.annotate('', xy=(axis_iteration_start_idx, latency_arrow_y), xytext=(latency_idx, latency_arrow_y), arrowprops=dict(arrowstyle="<->", color='#aaaaaa', lw=2))
ax.text((axis_iteration_start_idx + latency_idx) / 2, latency_text_y, f"Iter I+1: L = {single_iteration_latency}", color='#aaaaaa', ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

ax.annotate('', xy=(xmax + 1.0, bottom_y), xytext=(xmin, bottom_y), arrowprops=dict(arrowstyle="->", color='white', lw=1.5))
ax.text(xmax + 1.0, bottom_y + 0.18, 'Timeline (Hardware Cycle)', color='white', ha='right', va='bottom', fontsize=10)

legend_patches = [
    patches.Patch(facecolor=color_palette[prev_iteration_stage]['face'], edgecolor=color_palette[prev_iteration_stage]['edge'], label=f"Iteration I-1 [{stage_labels[prev_iteration_stage]}] Blocks"),
    patches.Patch(facecolor=color_palette[curr_iteration_stage]['face'], edgecolor=color_palette[curr_iteration_stage]['edge'], label=f"Iteration I [{stage_labels[curr_iteration_stage]}] Blocks"),
    patches.Patch(facecolor=color_palette[next_iteration_stage]['face'], edgecolor=color_palette[next_iteration_stage]['edge'], label=f"Iteration I+1 [{stage_labels[next_iteration_stage]}] Blocks"),
    patches.Patch(facecolor='#666666', edgecolor='none', alpha=issue_alpha, label='Dark segment = issue duration (d); light remainder = execution')
]
iteration_legend = ax.legend(handles=legend_patches, loc='upper right', facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)
ax.add_artist(iteration_legend)

wg_legend_patches = [
    patches.Patch(
        facecolor='#555555',
        edgecolor='#d8d8d8',
        hatch=wg_hatches.get(wgid, '---'),
        label=f"WG{wgid}: warpId // 4 == {wgid}",
    )
    for wgid in active_wgids
]
ax.legend(handles=wg_legend_patches, loc='upper right', bbox_to_anchor=(1.0, 0.82), facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)

plt.tight_layout()

# 保存生成图
output_filename = 'modulo_scheduling_same_line_with_desc.png'
plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"包含内部指令文字描述的同行流水图已成功导出: {output_filename}")


# ----------------- 7. 单个 II 窗口内的 op 启动顺序图 -----------------
# 这张图画的是 modulo steady state 的一个循环窗口：
# 横坐标按 slot = M % II 折叠，标签按 floor(M / II) 标出来自 i、i-1、i-2...
plt.close(fig)

single_items = []
single_events = {0, base_I, optimized_L}
for op, start_time in optimized_M.items():
    iter_distance = start_time // base_I
    slot_start = start_time % base_I
    slot_issue_end = slot_start + duration[op]
    slot_end = slot_start + latencies[op]
    single_items.append((op, slot_start, slot_end, slot_issue_end, iter_distance))
    single_events.update((slot_start, slot_issue_end, slot_end))

single_items.sort(key=lambda item: (item[1], item[4], item[0]))
single_events = sorted(single_events)
single_time_to_coord = {t: i for i, t in enumerate(single_events)}

def iter_label(distance):
    if distance == 0:
        return 'i'
    return f"i-{distance}"

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
ax_single.set_facecolor('#121212')
fig_single.patch.set_facecolor('#121212')

for op, slot_start, slot_end, slot_issue_end, iter_distance in single_items:
    y_pos = op_y_pos[op]
    cx_start = single_time_to_coord[slot_start]
    cx_end = single_time_to_coord[slot_end]
    cx_issue_end = single_time_to_coord[slot_issue_end]
    c_width = max(cx_end - cx_start, 0.25)
    issue_width = max(cx_issue_end - cx_start, 0.25)
    colors = single_color_palette.get(
        iter_distance,
        fallback_colors[iter_distance % len(fallback_colors)],
    )
    hatch = wg_hatches.get(op_wgid[op], '---')

    rect = patches.Rectangle(
        (cx_start, y_pos - box_height / 2),
        c_width, box_height,
        linewidth=1.5, edgecolor=colors['edge'], facecolor=colors['edge'],
        zorder=3, alpha=0.28
    )
    ax_single.add_patch(rect)

    issue_rect = patches.Rectangle(
        (cx_start, y_pos - box_height / 2),
        issue_width, box_height,
        linewidth=0, edgecolor='none', facecolor=colors['face'],
        zorder=4, alpha=issue_alpha
    )
    ax_single.add_patch(issue_rect)

    hatch_rect = patches.Rectangle(
        (cx_start, y_pos - box_height / 2),
        c_width, box_height,
        linewidth=0, edgecolor='#d8d8d8', facecolor='none',
        hatch=hatch, zorder=4.5, alpha=0.3
    )
    ax_single.add_patch(hatch_rect)

    label = f"{op_desc[op]} [{iter_label(iter_distance)}]"
    ax_single.text(
        cx_start + c_width / 2, y_pos, label,
        color='white', ha='center', va='center', fontsize=8,
        zorder=5, fontweight='bold', alpha=0.85
    )

for idx in range(len(single_events)):
    ax_single.axvline(x=idx, color='#2c2c2c', linestyle=':', linewidth=1.0, zorder=1)

for i in range(len(unit_boundaries) - 1):
    end_y = unit_boundaries[i][2]
    next_start_y = unit_boundaries[i + 1][1]
    sep_y = (end_y + next_start_y) / 2
    ax_single.axhline(y=sep_y, color='#555555', linestyle='--', linewidth=1.2, zorder=1)

ax_single.set_yticks(y_ticks_positions)
ax_single.set_yticklabels(y_ticks_labels, fontsize=9)

single_max_xtick_labels = 34
single_tick_step = max(1, int(np.ceil(len(single_events) / single_max_xtick_labels)))
single_important_times = {0, base_I, optimized_L}
single_xtick_positions = {
    idx
    for idx, t in enumerate(single_events)
    if t in single_important_times or idx % single_tick_step == 0
}
single_xtick_positions = sorted(single_xtick_positions)
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
ii_end_idx = single_time_to_coord[base_I]
range_arrow_y = single_bottom_y - 1.0
range_text_y = single_bottom_y - 0.9
ax_single.annotate('', xy=(ii_start_idx, range_arrow_y), xytext=(ii_end_idx, range_arrow_y), arrowprops=dict(arrowstyle="<->", color='#3a889e', lw=2))
ax_single.text((ii_start_idx + ii_end_idx) / 2, range_text_y, f"Single modulo window: II = {base_I}", color='#3a889e', ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

latency_idx = single_time_to_coord[optimized_L]
latency_arrow_y = single_bottom_y - 2.0
latency_text_y = single_bottom_y - 1.9
ax_single.annotate('', xy=(ii_start_idx, latency_arrow_y), xytext=(latency_idx, latency_arrow_y), arrowprops=dict(arrowstyle="<->", color='#aaaaaa', lw=2))
ax_single.text((ii_start_idx + latency_idx) / 2, latency_text_y, f"Logical iteration latency L = {optimized_L}", color='#aaaaaa', ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=6)

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
    patches.Patch(facecolor='#666666', edgecolor='none', alpha=issue_alpha, label='Dark segment = issue duration (d); light remainder = execution')
)
iter_legend = ax_single.legend(handles=single_iteration_legend, loc='upper right', facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)
ax_single.add_artist(iter_legend)

single_wg_legend = [
    patches.Patch(
        facecolor='#555555',
        edgecolor='#d8d8d8',
        hatch=wg_hatches.get(wgid, '---'),
        label=f"WG{wgid}: warpId // 4 == {wgid}",
    )
    for wgid in active_wgids
]
ax_single.legend(handles=single_wg_legend, loc='upper right', bbox_to_anchor=(1.0, 0.82), facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)

plt.tight_layout()
single_output_filename = 'single_iter_ops.png'
plt.savefig(single_output_filename, dpi=300, facecolor=fig_single.get_facecolor(), edgecolor='none')
print(f"单个 II 窗口内的 op 启动顺序图已成功导出: {single_output_filename}")
