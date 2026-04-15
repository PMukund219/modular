from std.runtime.asyncrt import DeviceContextPtr
from tensor import ManagedTensorSlice
from std.utils.index import IndexList


def _matmul_cpu(
    output: ManagedTensorSlice[mut=True, rank=2, ...],
    lhs: ManagedTensorSlice[dtype=output.dtype, rank=2, ...],
    rhs: ManagedTensorSlice[dtype=output.dtype, rank=2, ...],
    ctx: DeviceContextPtr,
):
    var M = output.dim_size(0)
    var N = output.dim_size(1)
    var K = lhs.dim_size(1)

    for m in range(M):
        for n in range(N):
            var acc = lhs.load[1](IndexList[2](m, 0)) * rhs.load[1](IndexList[2](0, n))
            for k in range(1, K):
                var lhs_idx = IndexList[2](m, k)
                var rhs_idx = IndexList[2](k, n)
                acc = acc + lhs.load[1](lhs_idx) * rhs.load[1](rhs_idx)

            output.store[1](IndexList[2](m, n), acc)