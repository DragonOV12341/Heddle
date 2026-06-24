import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 基础数据
I = 432
latencies = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 432, 6: 3, 7: 1, 
    8: 25, 9: 25, 10: 3, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 216
}
M = {0: 0, 1: 1, 2: 436, 3: 431, 4: 0, 5: 2, 6: 434, 7: 437, 8: 440, 9: 438, 10: 463, 11: 465, 12: 466, 13: 464, 14: 461, 15: 462, 16: 466}

fus = ["TMA", "TC", "ALU", "SFU"]
rrt_data = [
    {"slot": 0, "TMA": [0], "TC": [], "ALU": [4], "SFU": []},
    {"slot": 1, "TMA": [1], "TC": [1], "ALU": [1], "SFU": [1]},
    {"slot": 2, "TMA": [], "TC": [5], "ALU": [6], "SFU": []},
    {"slot": 4, "TMA": [], "TC": [], "ALU": [2], "SFU": []},
    {"slot": 5, "TMA": [], "TC": [], "ALU": [7], "SFU": []},
    {"slot": 6, "TMA": [], "TC": [], "ALU": [], "SFU": [9]},
    {"slot": 8, "TMA": [], "TC": [], "ALU": [], "SFU": [8]},
    {"slot": 29, "TMA": [14], "TC": [], "ALU": [], "SFU": []},
    {"slot": 30, "TMA": [15], "TC": [15], "ALU": [15], "SFU": [15]},
    {"slot": 31, "TMA": [], "TC": [], "ALU": [10], "SFU": []},
    {"slot": 32, "TMA": [], "TC": [], "ALU": [13], "SFU": []},
    {"slot": 33, "TMA": [], "TC": [], "ALU": [11], "SFU": []},
    {"slot": 34, "TMA": [], "TC": [16], "ALU": [12], "SFU": []},
]

# 2. 建立 FU -> Ops 的映射
fu_to_ops = {fu: [] for fu in fus}
op_info = {}
all_absolute_cycles = set()

for item in rrt_data:
    for fu in fus:
        for op in item[fu]:
            if op not in fu_to_ops[fu]:
                fu_to_ops[fu].append(op)
            abs_cycle = M[op]
            op_info[op] = {
                "start_cycle": abs_cycle, 
                "fu": fu, 
                "latency": latencies[op],
                "iter_id": abs_cycle // I
            }
            all_absolute_cycles.add(abs_cycle)

for fu in fus:
    fu_to_ops[fu].sort()

columns_flat = []
col_fu_parents = []  
for fu in fus:
    for op in fu_to_ops[fu]:
        columns_flat.append(f"op{op}")
        col_fu_parents.append(fu)

sorted_cycles = sorted(list(all_absolute_cycles))

# 【新逻辑】定义迭代（Iteration）的专属核心颜色
# Iter 0 分配蓝色系，Iter 1 分配绿色系，Iter 2 分配橙色系（以此类推）
iter_colors_base = {
    0: [0.12, 0.47, 0.71],  # 经典蓝 (RGB)
    1: [0.17, 0.63, 0.17],  # 经典绿 (RGB)
    2: [1.00, 0.50, 0.05],  # 橙色 (RGB)
}

# 3. 填充二维生命周期和状态矩阵
matrix_data = []
cell_types = []       # 记录是 'start', 'running' 还是 'empty'
cell_iter_ids = []    # 记录每个格子对应的迭代号，用于后面着色
row_labels = []

for current_cycle in sorted_cycles:
    row_labels.append(f"Cycle {current_cycle}")
    row_cells = []
    row_types = []
    row_iters = []
    
    for op_name in columns_flat:
        op = int(op_name.replace("op", ""))
        info = op_info[op]
        start = info["start_cycle"]
        latency = info["latency"]
        iter_id = info["iter_id"]
        
        if start == current_cycle:
            row_cells.append(f"★\nIter {iter_id}\n(L={latency})")
            row_types.append("start")
            row_iters.append(iter_id)
        elif start < current_cycle <= start + latency:
            row_cells.append(f"Iter {iter_id}\n│\n▼")
            row_types.append("running")
            row_iters.append(iter_id)
        else:
            row_cells.append("")
            row_types.append("empty")
            row_iters.append(-1)
            
    matrix_data.append(row_cells)
    cell_types.append(row_types)
    cell_iter_ids.append(row_iters)

# 4. 使用 Matplotlib 绘图
fig, ax = plt.subplots(figsize=(len(columns_flat) * 1.1 + 3, len(sorted_cycles) * 0.9 + 3))
ax.axis("off")

df = pd.DataFrame(matrix_data, columns=columns_flat, index=row_labels)
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    rowLabels=df.index,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 3.8)

# 5. 核心：遍历单元格修改属性，应用 Iteration 专属色
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)
    
    if row == 0:  # 子表头 (op名字栏)
        # 获取该 op 对应的发射 Iter ID，让表头颜色与它首次发射的迭代一致
        op_name = columns_flat[col]
        op = int(op_name.replace("op", ""))
        op_iter = op_info[op]["iter_id"]
        
        rgb = iter_colors_base.get(op_iter, [0.5, 0.5, 0.5])
        cell.set_facecolor(rgb + [0.85]) # 略带一点透明度
        cell.set_text_props(weight="bold", color="white")
        cell.set_edgecolor('black')
        
    elif col == -1:  # 行表头 (Cycle)
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#F2F2F2")
        cell.set_edgecolor('black')
        
    else:  # 内容数据区
        status = cell_types[row - 1][col]
        iter_id = cell_iter_ids[row - 1][col]
        
        if status != "empty" and iter_id in iter_colors_base:
            rgb = iter_colors_base[iter_id]
            if status == "start":
                # 发射状态：使用该迭代的深色（不透明）
                cell.set_facecolor(rgb + [1.0])  # RGBA
                cell.set_text_props(weight="bold", color="white")
            elif status == "running":
                # 延续状态：使用该迭代的同款颜色，但大幅降低不透明度（淡色流水条）
                cell.set_facecolor(rgb + [0.25])  # Alpha = 0.25 变淡
                cell.set_text_props(color="#222222", style="italic")
        else:
            cell.set_facecolor("#FFFFFF")
            
    # 如果当前列是某一个 FU 的“最后一列”，并且后面还有别的 FU，就把它的右侧线条加粗线强隔离
    if col >= 0 and col < len(col_fu_parents) - 1:
        current_fu = col_fu_parents[col]
        next_fu = col_fu_parents[col + 1]
        if current_fu != next_fu:
            cell.set_edgecolor('black')
            cell.set_linewidth(1.8)

# 6. 在图形上方手动绘制大表头 (FU 名字栏)
cells = table.get_celld()
fu_bounds = {}
for idx, fu in enumerate(col_fu_parents):
    if fu not in fu_bounds:
        fu_bounds[fu] = [idx, idx]
    else:
        fu_bounds[fu][1] = idx

for fu, (start_idx, end_idx) in fu_bounds.items():
    total_cols = len(columns_flat)
    start_pos = (start_idx) / total_cols
    end_pos = (end_idx + 1) / total_cols
    
    ax.text(
        (start_pos + end_pos) / 2 - 0.02, 0.93, fu,
        transform=ax.transAxes,
        fontsize=12, weight="bold", color="white",
        bbox=dict(facecolor='#2F4F4F', edgecolor='black', boxstyle='round,pad=0.4'),
        ha='center'
    )

plt.title(
    "Modulo Scheduling: Iteration-Based Color Coding (Overlap View)",
    fontsize=15,
    weight="bold",
    pad=50
)
plt.tight_layout()
plt.savefig("modulo_iteration_colors.png", bbox_inches="tight", dpi=300)
plt.show()