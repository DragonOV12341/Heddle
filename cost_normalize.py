from typing import List, Tuple
from ortools.sat.python import cp_model

def cost_normalize(C: List[int], U: int = 300) -> Tuple[List[int], int]:
    """
    实现论文 5.2 节中的 Cost Normalization 算法。
    使用 Google OR-Tools CP-SAT 求解器将原始的大 cycle counts 转换为
    比例尽可能接近、数值尽可能小的整型 cycle counts。
    
    :param C: 原始的 cycle 计数列表 (C[i] > 0)
    :param U: 归一化后数值的上限 (论文默认 U = 300)
    :return: (C_prime, F) 
             C_prime: 归一化后的整型 cycle 列表
             F: 最小化比例偏差后的目标值
    """
    n = len(C)
    model = cp_model.CpModel()
    
    # 1. 声明决策变量
    # C'[i] 的取值范围约束在 [1, U] 之间，避免全 0 退化解
    C_prime = [model.NewIntVar(1, U, f"C_prime_{i}") for i in range(n)]
    
    # F 代表比例偏差的最大值。偏差为非负数，其最大可能取值为 U * max(C)
    max_c = max(C) if C else 1
    F = model.NewIntVar(0, U * max_c, "F")
    
    # 2. 添加约束条件：-F <= C[i] * C'[j] - C[j] * C'[i] <= F
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # 计算线性组合：C[i] * C'[j] - C[j] * C'[i]
            expr = C[i] * C_prime[j] - C[j] * C_prime[i]
            
            # 约束：expr <= F 且 expr >= -F
            model.Add(expr <= F)
            model.Add(expr >= -F)
            
    # 3. 设置优化目标：最小化偏差 F
    model.Minimize(F)
    
    # 4. 调用求解器
    solver = cp_model.CpSolver()
    # 限制求解时间（例如最高 10 秒，通常几毫秒即可解出）
    solver.parameters.max_time_in_seconds = 10.0
    
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        normalized_costs = [solver.Value(C_prime[i]) for i in range(n)]
        min_F = solver.Value(F)
        return normalized_costs, min_F
    else:
        raise RuntimeError("未能找到可行解，请检查输入或调大上限 U。")

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟 Hopper 架构下的 Tile-level 耗时（原始周期数值极大）
    # 假设 GEMM 约 1000 周期，EXP 约 350 周期，ADD 约 40 周期
    original_cycles = []
    # original_cycles = [1000, 350, 40]
    latencies={0: 280, 1: 1, 2: 1, 3: 1, 4: 1, 5: 152, 6: 3, 7: 1, 8: 25, 9: 25, 10: 3, 11: 1, 12: 1, 13: 1, 14: 280, 15: 1, 16: 176}
    for k,v in latencies.items() :
        original_cycles.append(v)
        
    print(f"原始 Cycle 计数: {original_cycles}")
    try:
        normalized, final_F = cost_normalize(original_cycles, U=300)
        print(f"归一化后的 Cycle (U=300): {normalized}")
        print(f"最优偏差 F: {final_F}")
        
        # 验证比例
        print("\n比例对比（原始 vs 归一化）:")
        for i in range(len(original_cycles)):
            for j in range(i+1, len(original_cycles)):
                orig_ratio = original_cycles[i] / original_cycles[j]
                norm_ratio = normalized[i] / normalized[j]
                print(f"Index ({i}, {j}) -> 原始比: {orig_ratio:.4f} | 归一化比: {norm_ratio:.4f}")
                
    except Exception as e:
        print(f"求解失败: {e}")