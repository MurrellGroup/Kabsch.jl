module KabschCUDAExt

using Kabsch
using CUDA

using LinearAlgebra

function Kabsch.batched_svd(A::CuArray{<:AbstractFloat,3})
    @assert size(A, 1) == 3 && size(A, 2) == 3
    U, _, V = svd(copy(A))
    return U, V
end

function Kabsch.batched_det(A::CuArray{<:AbstractFloat,3})
    @assert size(A, 1) == 3 && size(A, 2) == 3
    function kernel!(out, A)
        k = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
        if k ≤ size(A, 3)
            a11 = A[1,1,k]; a12 = A[1,2,k]; a13 = A[1,3,k]
            a21 = A[2,1,k]; a22 = A[2,2,k]; a23 = A[2,3,k]
            a31 = A[3,1,k]; a32 = A[3,2,k]; a33 = A[3,3,k]
            out[k] = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
        end
        return
    end
    n = size(A, 3)
    out = similar(A, n)
    kernel! = CUDA.@cuda launch=false kernel!(out, A)
    config = CUDA.launch_configuration(kernel!.fun)
    threads = min(n, config.threads)
    kernel!(out, A; threads, blocks = cld(n, threads))
    out
end

end
