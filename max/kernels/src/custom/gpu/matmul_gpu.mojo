from std.math import ceildiv
from std.gpu import (
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
)
from std.runtime.asyncrt import DeviceContextPtr
from tensor import ManagedTensorSlice
from std.utils.index import IndexList


def _matmul_gpu(
    output: ManagedTensorSlice[mut=True, rank=2, ...],
    lhs: ManagedTensorSlice[dtype=output.dtype, rank=2, ...],
    rhs: ManagedTensorSlice[dtype=output.dtype, rank=2, ...],
    ctx: DeviceContextPtr,
) raises:


    comptime BLOCK_SIZE = 16  

    var gpu_ctx = ctx.get_device_context()
    var M = output.dim_size(0)
    var N = output.dim_size(1)
    var K = lhs.dim_size(1)

    @parameter
    def matmul_kernel(M: Int, N: Int, K: Int):
        var col = Int(block_dim.x * block_idx.x + thread_idx.x)
        var row = Int(block_dim.y * block_idx.y + thread_idx.y)

        if row >= M or col >= N:
            return

        var acc = lhs.load[1](IndexList[2](row, 0)) * rhs.load[1](IndexList[2](0, col))
        for k in range(1, K):
            acc = acc + lhs.load[1](IndexList[2](row, k)) * rhs.load[1](IndexList[2](k, col))

        output.store[1](IndexList[2](row, col), acc)

    var grid_x = ceildiv(N, BLOCK_SIZE)
    var grid_y = ceildiv(M, BLOCK_SIZE)

    gpu_ctx.enqueue_function_experimental[matmul_kernel](
        M, N, K,
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )