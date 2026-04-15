import compiler
from std.math import ceildiv, max, min
from std.gpu import (
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
)
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, ManagedTensorSlice
from std.utils.index import IndexList

def _conv2d_cpu(
    output: ManagedTensorSlice[mut=True, rank=4, ...],
    input:  ManagedTensorSlice[dtype=output.dtype, rank=4, ...],
    weight: ManagedTensorSlice[dtype=output.dtype, rank=4, ...],
    ctx: DeviceContextPtr,
):
    var N    = output.dim_size(0)
    var Cout = output.dim_size(1)
    var Hout = output.dim_size(2)
    var Wout = output.dim_size(3)
    var Cin  = input.dim_size(1)
    var KH   = weight.dim_size(2)
    var KW   = weight.dim_size(3)

    for n in range(N):
        for co in range(Cout):
            for oh in range(Hout):
                for ow in range(Wout):
                    var acc = Scalar[output.dtype](0)
                    for ci in range(Cin):
                        for kh in range(KH):
                            for kw in range(KW):
                                var ih = oh + kh
                                var iw = ow + kw
                                var v = input.load[1](IndexList[4](n, ci, ih, iw))
                                var w = weight.load[1](IndexList[4](co, ci, kh, kw))
                                acc = acc + v * w

                    output.store[1](IndexList[4](n, co, oh, ow), acc)



@compiler.register("convolution2d")
struct Conv2d:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=4, ...],
        input:  InputTensor[dtype=output.dtype, rank=4, ...],
        weight: InputTensor[dtype=output.dtype, rank=4, ...],
        ctx: DeviceContextPtr,
    ) raises:

        debug_assert(
            output.dim_size(0) == input.dim_size(0),
            "conv2d: batch size mismatch"
        )
        debug_assert(
            weight.dim_size(1) == input.dim_size(1),
            "conv2d: weight C_in must match input C_in"
        )
        debug_assert(
            output.dim_size(1) == weight.dim_size(0),
            "conv2d: output C_out must match weight C_out"
        )
        debug_assert(
            output.dim_size(2) == input.dim_size(2) - weight.dim_size(2) + 1,
            "conv2d: H_out must equal H - KH + 1 (no padding, stride=1)"
        )
        debug_assert(
            output.dim_size(3) == input.dim_size(3) - weight.dim_size(3) + 1,
            "conv2d: W_out must equal W - KW + 1 (no padding, stride=1)"
        )

        comptime if target == "cpu":
            _conv2d_cpu(output, input, weight, ctx)
        # elif target == "gpu":
        #     _conv2d_gpu(output, input, weight, ctx)
        else:
            raise Error("No known target: ", target)



# resources 
#https://medium.com/@abhishekjainindore24/all-about-convolutions-kernels-features-in-cnn-c656616390a1
#https://www.geeksforgeeks.org/computer-vision/apply-a-2d-convolution-operation-in-pytorch/
#https://ulhpc-tutorials.readthedocs.io/en/latest/cuda/exercises/convolution/
#