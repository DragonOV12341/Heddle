import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ----------------- 1. 原始数据与配置 -----------------
base_I = 432
optimized_M = {
    0: 0, 1: 1, 2: 4, 3: 6, 4: 0, 5: 2, 6: 434, 7: 437, 
    8: 439, 9: 438, 10: 463, 11: 465, 12: 466, 13: 464, 
    14: 2, 15: 3, 16: 466
}

latencies = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 432, 6: 3, 7: 1, 
    8: 25, 9: 25, 10: 3, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 216
}

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
    'TC': [5, 16],
    'ALU': [2, 3, 4, 6, 7, 10, 11, 12, 13],
    'SFU': [8, 9]
}

# ----------------- 2. 生成双迭代全操作时间节点 -----------------
plot_items = []
all_events = set()

all_events.add(0)
all_events.add(base_I)      # 432 周期分界
all_events.add(base_I * 2)  # 864 周期分界

# A. 铺平迭代 I-1 的全部分量（0 ~ 432 周期内投影）
for op, start_time in optimized_M.items():
    lat = latencies[op]
    t_start = start_time % base_I
    t_end = t_start + lat
    plot_items.append((op, t_start, t_end, 1)) # 1 代表前代 [i-1]
    all_events.add(t_start)
    all_events.add(t_end)

# B. 铺平迭代 I 的全部分量（整体向后平移一个 II 周期：432 ~ 864 内投影）
for op, start_time in optimized_M.items():
    lat = latencies[op]
    t_start = (start_time % base_I) + base_I
    t_end = t_start + lat
    plot_items.append((op, t_start, t_end, 0)) # 0 代表当代 [i]
    all_events.add(t_start)
    all_events.add(t_end)

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
fig, ax = plt.subplots(figsize=(20, 10), dpi=150)
ax.set_facecolor('#121212')
fig.patch.set_facecolor('#121212')

box_height = 0.55
color_palette = {
    1: {'face': '#5c3d2e', 'edge': '#ba7a5f'},  # 迭代 I-1 [i-1] -> 深棕
    0: {'face': '#1f4e5b', 'edge': '#3a889e'},  # 迭代 I [i]     -> 深青
}

for op, start, end, stage in plot_items:
    y_pos = op_y_pos[op] # 确保同一 op 在同一行
    
    cx_start = time_to_coord[start]
    cx_end = time_to_coord[end]
    c_width = cx_end - cx_start
    
    colors = color_palette[stage]
    
    rect = patches.Rectangle(
        (cx_start, y_pos - box_height / 2), 
        c_width, box_height, 
        linewidth=1.5, edgecolor=colors['edge'], facecolor=colors['face'], zorder=3
    )
    ax.add_patch(rect)
    
    # 【核心恢复】将完整的算子指令描述 (desc) 加上迭代标签写进滑块内
    desc = op_desc[op]
    label = f"{desc} [i-1]" if stage == 1 else f"{desc} [i]"
    ax.text(
        cx_start + c_width / 2, y_pos, label, 
        color='white', ha='center', va='center', fontsize=8, zorder=4, fontweight='bold'
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

ax.set_xticks(range(len(sorted_events)))
ax.set_xticklabels([str(t) for t in sorted_events], fontsize=7, rotation=45)

# ----------------- 6. 底部时轴与架构参数标注（II 与 L） -----------------
bottom_y = -1.0
ax.spines['bottom'].set_position(('data', bottom_y))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

xmin, xmax = -0.5, len(sorted_events) - 0.5
ax.set_xlim(xmin, xmax + 1.5)
ax.set_ylim(bottom_y - 2.5, total_rows - 0.2)

idx_0 = time_to_coord[0]
idx_432 = time_to_coord[432]
idx_864 = time_to_coord[864]

# A. 标注迭代 I-1 的触发窗口范围 (0 ~ 432)
ax.annotate('', xy=(idx_0, bottom_y - 0.6), xytext=(idx_432, bottom_y - 0.6), arrowprops=dict(arrowstyle="<->", color='#ba7a5f', lw=2))
ax.text((idx_0 + idx_432)/2, bottom_y - 0.5, 'Iteration I-1 Range (II = 432)', color='#ba7a5f', ha='center', va='bottom', fontsize=11, fontweight='bold')

# B. 标注迭代 I 的触发窗口范围 (432 ~ 864)
ax.annotate('', xy=(idx_432, bottom_y - 0.6), xytext=(idx_864, bottom_y - 0.6), arrowprops=dict(arrowstyle="<->", color='#3a889e', lw=2))
ax.text((idx_432 + idx_864)/2, bottom_y - 0.5, 'Iteration I Range (II = 432)', color='#3a889e', ha='center', va='bottom', fontsize=11, fontweight='bold')

# C. 标注单个核心循环的完整处理生命周期 Latency L (0 到 682)
idx_682 = time_to_coord[682]
ax.annotate('', xy=(idx_0, bottom_y - 1.5), xytext=(idx_682, bottom_y - 1.5), arrowprops=dict(arrowstyle="<->", color='#aaaaaa', lw=2))
ax.text((idx_0 + idx_682)/2, bottom_y - 1.4, 'Single Iteration Latency (L = 682)', color='#aaaaaa', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.annotate('', xy=(xmax + 1.0, bottom_y), xytext=(xmin, bottom_y), arrowprops=dict(arrowstyle="->", color='white', lw=1.5))
ax.text(xmax + 1.0, bottom_y - 0.3, 'Timeline (Hardware Cycle)', color='white', ha='right', fontsize=10)

legend_patches = [
    patches.Patch(facecolor=color_palette[1]['face'], edgecolor=color_palette[1]['edge'], label='Iteration I-1 [i-1] Blocks'),
    patches.Patch(facecolor=color_palette[0]['face'], edgecolor=color_palette[0]['edge'], label='Iteration I [i] Blocks')
]
ax.legend(handles=legend_patches, loc='upper right', facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)

plt.tight_layout()

# 保存生成图
output_filename = 'modulo_scheduling_same_line_with_desc.png'
plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"包含内部指令文字描述的同行流水图已成功导出: {output_filename}")