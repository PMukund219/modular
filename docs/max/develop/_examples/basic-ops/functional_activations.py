# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# DOC: max/develop/basic-ops.mdx

import max.experimental.functional as F
from max.experimental.tensor import Tensor
from pathlib import Path
from max.dtype import DType
from max.driver import CPU , Accelerator



mojo_kernel_path = Path(__file__).parent.parent.parent.parent.parent.parent / "max" / "kernels" / "src" / "custom"


a = Tensor([[1.0, 2.0, 3.0]], dtype=DType.float32, device=Accelerator())
b = Tensor([[4.0], [5.0], [6.0]], dtype=DType.float32, device=Accelerator())
result = F.custom(
    "matmul2",
    device=a.device,
    values=[a, b],
    out_types=[a.type],
    custom_extensions=mojo_kernel_path
)[0]

print(f"Custom Matrix Multiplication Result: {result}")
