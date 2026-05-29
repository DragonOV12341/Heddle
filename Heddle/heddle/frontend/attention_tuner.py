# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class HardwareConstraints:
    sram_size: int = 108 * 1024 # 108KB for A100 (configurable)
    dtype_bytes: int = 2 # FP16

@dataclass
class TilingConfig:
    block_q: int
    block_kv: int
    num_stages: int
    score_sram_usage: int
    kv_sram_usage: int
    total_sram_usage: int
    occupancy: float

class AttentionTuner:
    """
    Auto-tuner that searches for optimal FlashAttention tiling configurations
    based on hardware constraints (SRAM size).
    """
    def __init__(self, head_dim: int, constraints: HardwareConstraints = HardwareConstraints()):
        self.head_dim = head_dim
        self.constraints = constraints
        
    def search(self) -> List[TilingConfig]:
        """
        Enumerate valid tiling configurations that fit in SRAM.
        """
        valid_configs = []
        
        # Search space
        block_q_candidates = [32, 64, 128, 256]
        block_kv_candidates = [32, 64, 128, 256]
        stages_candidates = [1, 2, 3, 4]
        
        print(f"[AttentionTuner] Searching tiling for HeadDim={self.head_dim}...")
        
        for bq in block_q_candidates:
            for bkv in block_kv_candidates:
                for stages in stages_candidates:
                    config = self._evaluate_config(bq, bkv, stages)
                    if config:
                        valid_configs.append(config)
        
        # Sort by heuristic: larger blocks usually better for Tensor Core utilization
        # But we also need to balance occupancy.
        # Simple heuristic: maximize block_q * block_kv (throughput)
        valid_configs.sort(key=lambda c: c.block_q * c.block_kv, reverse=True)
        return valid_configs

    def _evaluate_config(self, bq, bkv, stages) -> TilingConfig:
        d = self.head_dim
        dtype = self.constraints.dtype_bytes
        
        # Calculate SRAM usage for FlashAttention Pattern
        # 1. Q Block: [Block_Q, Dim]
        #    Usually double buffered or just 1 buffer if loaded in outer loop
        size_q = bq * d * dtype
        
        # 2. K, V Blocks: [Block_KV, Dim] * Stages (Pipeline)
        size_k = bkv * d * dtype * stages
        size_v = bkv * d * dtype * stages
        
        # 3. Score Block (Accumulator): [Block_Q, Block_KV]
        #    Usually FP32 accumulation
        size_score = bq * bkv * 4 # FP32
        
        # 4. Softmax buffers (Row Max, Sum, etc.): [Block_Q]
        size_softmax = bq * 4 * 2 # Max + Sum
        
        total_usage = size_q + size_k + size_v + size_score + size_softmax
        
        if total_usage > self.constraints.sram_size:
            return None
            
        # Calculate Occupancy (Simulated)
        # Max SRAM usage per SM is limit.
        # How many blocks can fit?
        active_blocks = self.constraints.sram_size // total_usage
        occupancy = min(active_blocks * 4 * 32, 2048) / 2048 # Rough occupancy metric
        
        return TilingConfig(
            block_q=bq, 
            block_kv=bkv, 
            num_stages=stages,
            score_sram_usage=size_score,
            kv_sram_usage=size_k + size_v,
            total_sram_usage=total_usage,
            occupancy=occupancy
        )

if __name__ == "__main__":
    # Demo: Search for standard settings
    tuner = AttentionTuner(head_dim=64)
    configs = tuner.search()
    
    print(f"\nFound {len(configs)} valid configs. Top 5:")
    for i, c in enumerate(configs[:5]):
        print(f"Rank {i+1}: BlockQ={c.block_q}, BlockKV={c.block_kv}, Stages={c.num_stages}")
        print(f"       SRAM={c.total_sram_usage/1024:.1f}KB (Score: {c.score_sram_usage/1024:.1f}KB)")
        
    print("\n[Insight] Note how BlockKV is often smaller or equal to BlockQ to fit KV pipeline stages.")

