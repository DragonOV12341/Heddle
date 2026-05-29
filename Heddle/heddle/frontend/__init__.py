# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
TileLang Compiler Frontend.

This module provides integration with PyTorch's compilation stack (torch.compile)
and implements the frontend logic for translating high-level deep learning graphs
into optimized TileLang kernels.

Architecture:
-------------
1. **Frontend**:
   - Uses `torch.compile` + `torch.fx` to capture the user's computation graph.
   - Converts PyTorch FX graphs into an intermediate representation suitable for analysis.

2. **Middle-end**:
   - **Pattern Matching**: Identifies well-known high-performance operator patterns (e.g., FlashAttention, GEMM)
     and replaces them with highly optimized TileLang templates.
   - **Codegen**: For unknown or custom pointwise/elementwise subgraphs, uses the `ElementwiseCodegen`
     transpiler to automatically generate TileLang source code (TIR).

3. **Backend**:
   - Invokes the TileLang JIT compiler.
   - Lowers the generated Python/TIR code to TVM TIR, and finally to machine code (e.g., CUDA PTX).
"""

from .backend import tilelang_backend
from . import patterns
from . import codegen
