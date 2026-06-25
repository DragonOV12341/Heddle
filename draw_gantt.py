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

# 引入你的操作具体描述数据
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

# ----------------- 2. 构建非均匀时间轴映射 -----------------
all_time_events = set()
for op, start_time in optimized_M.items():
    x_start = start_time % base_I
    lat = latencies[op]
    x_end = x_start + lat
    all_time_events.add(x_start)
    all_time_events.add(x_end)

sorted_events = sorted(list(all_time_events))
time_to_coord = {t: i for i, t in enumerate(sorted_events)}

# ----------------- 3. 计算 Y 轴排布（独占一行并引入描述） -----------------
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
        
        start_time = optimized_M[op]
        stage = start_time // base_I
        desc = op_desc[op]
        
        # 将左侧坐标刻度的标签也更新为 Description 形式
        lbl = f"s{op}: {desc} [i]" if stage == 0 else f"s{op}: {desc} [i-{stage}]"
        y_ticks_labels.append(f"{unit}: {lbl}")
        
        current_y += 1
        
    end_y_for_unit = current_y - 1
    unit_boundaries.append((unit, start_y_for_unit, end_y_for_unit))
    current_y += 0.6 

total_rows = current_y

# ----------------- 4. 绘图基础配置 -----------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(18, 10), dpi=150) # 调宽到18以给长字符串留出空间
ax.set_facecolor('#121212')
fig.patch.set_facecolor('#121212')

box_height = 0.55 

# 迭代颜色配置
color_palette = {
    0: {'face': '#1f4e5b', 'edge': '#3a889e'},  # 迭代 [i]
    1: {'face': '#5c3d2e', 'edge': '#ba7a5f'},  # 迭代 [i-1]
}
default_color = {'face': '#3d3d3d', 'edge': '#888888'}

# ----------------- 5. 使用虚拟坐标绘制各模块 -----------------
for op, start_time in optimized_M.items():
    if op not in op_y_pos:
        continue
        
    y_pos = op_y_pos[op]
    x_start_val = start_time % base_I
    lat = latencies[op]
    x_end_val = x_start_val + lat
    stage = start_time // base_I
    
    cx_start = time_to_coord[x_start_val]
    cx_end = time_to_coord[x_end_val]
    c_width = cx_end - cx_start 
    
    colors = color_palette.get(stage, default_color)
    
    rect = patches.Rectangle(
        (cx_start, y_pos - box_height / 2), 
        c_width, box_height, 
        linewidth=1.5, edgecolor=colors['edge'], facecolor=colors['face'], zorder=3
    )
    ax.add_patch(rect)
    
    # 【核心修改】：获取具体的 description 并保留 [i]/[i-1] 迭代标记
    desc = op_desc[op]
    label = f"{desc} [i]" if stage == 0 else f"{desc} [i-{stage}]"
    
    ax.text(
        cx_start + c_width / 2, y_pos, label, 
        color='white', ha='center', va='center', fontsize=8, zorder=4, fontweight='bold'
    )

# ----------------- 6. 线条与轴线修饰 -----------------
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
ax.set_xticklabels([str(t) for t in sorted_events], fontsize=8, rotation=45) 

bottom_y = -1.0
ax.spines['bottom'].set_position(('data', bottom_y))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

xmin, xmax = -0.5, len(sorted_events) - 0.5
ax.set_xlim(xmin, xmax + 1.5)
ax.set_ylim(bottom_y - 0.5, total_rows - 0.2)

ax.annotate('', xy=(xmax + 1.0, bottom_y), xytext=(xmin, bottom_y),
            arrowprops=dict(arrowstyle="->", color='white', lw=1.5))
ax.text(xmax + 1.0, bottom_y - 0.6, 'clock-cycle (mod 432, Non-uniform)', color='white', ha='right', fontsize=10)

legend_patches = [
    patches.Patch(facecolor=color_palette[0]['face'], edgecolor=color_palette[0]['edge'], label='Iteration [i]'),
    patches.Patch(facecolor=color_palette[1]['face'], edgecolor=color_palette[1]['edge'], label='Iteration [i-1]')
]
ax.legend(handles=legend_patches, loc='upper right', facecolor='#1A1A1A', edgecolor='#444444', fontsize=10)

plt.tight_layout()

# ----------------- 7. 保存高保真图 -----------------
output_filename = 'modulo_scheduling_gantt_desc.png'
plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"包含详细描述的高级调度甘特图已保存至: {output_filename}")