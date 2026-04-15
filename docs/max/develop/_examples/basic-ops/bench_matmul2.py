from pathlib import Path
import time
import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

MOJO_KERNELS = Path(__file__).parent.parent.parent.parent.parent.parent / "max" / "kernels" / "src" / "custom"
DTYPE = DType.float32


def build_model(device, M, K, N):
    dev = DeviceRef.from_device(device)
    graph = Graph(
        "matmul2",
        forward=lambda a, b: ops.custom(
            name="matmul2",
            device=dev,
            values=[a, b],
            out_types=[TensorType(dtype=DTYPE, shape=[M, N], device=dev)],
        )[0].tensor,
        input_types=[
            TensorType(DTYPE, shape=[M, K], device=dev),
            TensorType(DTYPE, shape=[K, N], device=dev),
        ],
        custom_extensions=[MOJO_KERNELS],
    )
    return InferenceSession(devices=[device]).load(graph)


def bench(device, M, K, N):
    model = build_model(device, M, K, N)

    a_np = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
    b_np = np.random.uniform(-1, 1, (K, N)).astype(np.float32)
    a = Buffer.from_numpy(a_np).to(device)
    b = Buffer.from_numpy(b_np).to(device)

    # warmup
    model.execute(a, b)[0].to(CPU()).to_numpy()

    st        = time.time()
    result_np = model.execute(a, b)[0].to(CPU()).to_numpy()
    et        = time.time()
    ms        = (et - st) * 1e3

    passed = np.allclose(result_np, a_np @ b_np, atol=1e-3)
    gflops = (2 * M * N * K) / (ms * 1e-3) / 1e9
    return passed, ms, gflops


if __name__ == "__main__":
    cpu = CPU()
    gpu = Accelerator() if accelerator_count() > 0 else None

    shapes = [
        (64,   64,   64),
        (256,  256,  256),
        (512,  512,  512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    if gpu:
        print(f"{'M':>6} {'K':>6} {'N':>6}  {'CPU ms':>8}  {'CPU GFLOPS':>10}  {'GPU ms':>8}  {'GPU GFLOPS':>10}  {'Speedup':>8}")
        print("-" * 76)
        for M, K, N in shapes:
            cpu_ok, cpu_ms, cpu_gf = bench(cpu, M, K, N)
            gpu_ok, gpu_ms, gpu_gf = bench(gpu, M, K, N)
            status = "PASS" if (cpu_ok and gpu_ok) else "FAIL"
            print(f"{M:>6} {K:>6} {N:>6}  {cpu_ms:>8.3f}  {cpu_gf:>10.2f}  {gpu_ms:>8.3f}  {gpu_gf:>10.2f}  {cpu_ms/gpu_ms:>7.2f}x  {status}")
    else:
        print(f"{'M':>6} {'K':>6} {'N':>6}  {'CPU ms':>8}  {'GFLOPS':>8}  {'Result':>6}")
        print("-" * 44)
        for M, K, N in shapes:
            ok, ms, gf = bench(cpu, M, K, N)
            print(f"{M:>6} {K:>6} {N:>6}  {ms:>8.3f}  {gf:>8.2f}  {'PASS' if ok else 'FAIL'}")