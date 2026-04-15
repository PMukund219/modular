import max.experimental.functional as F
from max.experimental.tensor import Tensor
from pathlib import Path
from max.dtype import DType
from max.driver import CPU, Accelerator
from max.experimental.tensor import TensorType  

mojo_kernel_path = Path(__file__).parent.parent.parent.parent.parent.parent / "max" / "kernels" / "src" / "custom"

device = CPU()

input_tensor = Tensor(
    [[[[1.0, 2.0, 3.0, 4.0, 5.0],
       [6.0, 7.0, 8.0, 9.0, 10.0],
       [11.0, 12.0, 13.0, 14.0, 15.0],
       [16.0, 17.0, 18.0, 19.0, 20.0],
       [21.0, 22.0, 23.0, 24.0, 25.0]]]],
    dtype=DType.float32,
    device=device,
)

weight_tensor = Tensor(
    [[[[1.0,  0.0, -1.0],
       [2.0,  0.0, -2.0],
       [1.0,  0.0, -1.0]]]],
    dtype=DType.float32,
    device=device,
)

output_type = TensorType(DType.float32, [1, 1, 3, 3], device=device)

result = F.custom(
    "convolution2d",
    device=device,
    values=[input_tensor, weight_tensor],
    out_types=[output_type],
    custom_extensions=mojo_kernel_path,
)[0]

print(f"Input shape:  {input_tensor.shape}")
print(f"Weight shape: {weight_tensor.shape}")
print(f"Output shape: {result.shape}")
print(f"Conv2D Result:\n{result}")