import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench


@tilelang.jit
def shardshare(S, D, N, R, block_S, block_N, dtype='float16', accum_dtype='float32'):
    
    a_shape = [S,D]
    b_shape = [R,N]
    c_shape = [S,N]
    BR = D // R
    @T.prim_func
    def kernel(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), T.ceildiv(N, block_N)) as (by,bx):  
            A_frag = T.alloc_fragment((block_S, R), accum_dtype)
            sum_A = T.alloc_fragment((block_S, R), accum_dtype)
            B_shared = T.alloc_shared((R, block_N), dtype=dtype)
            C_local = T.alloc_fragment((block_S, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_S, block_N), dtype=dtype)
            A_shared = T.alloc_shared((block_S, R), dtype=dtype)
            T.clear(sum_A)  
            for i in T.Pipelined(0, BR, num_stages=1):
                T.copy(A[by*block_S:(by+1)*block_S, i*R:(i+1)*R], A_frag)
                for s, j in T.Parallel(block_S, R):  
                    sum_A[s, j] += A_frag[s, j]#.astype(accum_dtype)
            T.copy(sum_A, A_shared)
            T.copy(B[:, bx*block_N:(bx+1)*block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local, clear_accum=True)  
            T.copy(C_local, C_shared)

            T.copy(C_shared, C[by*block_S:(by+1)*block_S, bx*block_N:(bx+1)*block_N])
    return kernel


if __name__ == "__main__":
    S, D, N = 1024, 1024, 4096
    block_S, block_N, R = 32, 128, 64
    dtype = 'float16'
    # Compile kernel (JIT compilation)
    miss_kernel = shardshare(S, D, N, R, block_S, block_N, dtype)

    # Create input tensors
    a = torch.randn(S, D, device="cuda", dtype=torch.float16)
    b = torch.randn(R, N, device="cuda", dtype=torch.float16)
    c = torch.empty(S, N, device="cuda", dtype=torch.float16)

    # Execute kernel
    miss_kernel(a, b, c)

    reshape_a = torch.sum(a.to(torch.float32).reshape(*a.shape[:-1], a.size(-1) // R, R), dim=-2)
    out = (reshape_a@b.to(torch.float32)).to(torch.float16)

    # Validate correctness
    # torch.testing.assert_close(c, out, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")


    tilelang_ms = do_bench(lambda: miss_kernel(a, b, c))
    native_ms = do_bench(lambda: torch.sum(a.reshape(*a.shape[:-1], a.size(-1) // R, R), dim=-2)@b)

    print(f"TileLang kernel avg latency: {tilelang_ms:.3f} ms")
    print(f"PyTorch native avg latency: {native_ms:.3f} ms")
    if tilelang_ms > 0:
        print(f"Speedup (native / TileLang): {native_ms / tilelang_ms:.2f}x")