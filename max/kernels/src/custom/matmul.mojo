# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

from .cpu.matmul_cpu import _matmul_cpu
from .gpu.matmul_gpu import _matmul_gpu


@compiler.register("matmul2")
struct Matmul:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2, ...],
        lhs: InputTensor[dtype=output.dtype, rank=2, ...],
        rhs: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:

        debug_assert(
            lhs.dim_size(1) == rhs.dim_size(0),
            "matmul: lhs columns must equal rhs rows (K dimension mismatch)"
        )
        debug_assert(
            output.dim_size(0) == lhs.dim_size(0),
            "matmul: output rows must equal lhs rows (M dimension mismatch)"
        )
        debug_assert(
            output.dim_size(1) == rhs.dim_size(1),
            "matmul: output columns must equal rhs columns (N dimension mismatch)"
        )

        
        comptime if target == "cpu":
            _matmul_cpu(output, lhs, rhs, ctx)
        elif target == "gpu":
            _matmul_gpu(output, lhs, rhs, ctx)
        else:
            raise Error("No known target: ", target)