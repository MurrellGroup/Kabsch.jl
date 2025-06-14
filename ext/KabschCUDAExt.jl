module KabschCUDAExt

using Kabsch
using CUDA

using LinearAlgebra

function Kabsch.batched_svd(A::CuArray{<:AbstractFloat,3})
    @assert size(A, 1) == 3 && size(A, 2) == 3
    U, _, V = svd(copy(A))
    return U, V
end

# taken from https://github.com/MurrellGroup/RotationMap.jl/blob/main/src/RotationMap.jl
# written by bicycle1885
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

function Kabsch.rmsd(P::CuArray{<:AbstractFloat,3}, Q::CuArray{<:AbstractFloat,3})
    size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    
    # Simple 2D kernel: each thread handles one point in one batch
    function kernel_2d!(partial_sums, P, Q, n_dims, n_points, n_batches)
        batch_idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
        point_idx = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
        
        if batch_idx ≤ n_batches && point_idx ≤ n_points
            sum_sq = zero(eltype(P))
            for d in 1:n_dims
                diff = P[d, point_idx, batch_idx] - Q[d, point_idx, batch_idx]
                sum_sq += diff * diff
            end
            partial_sums[point_idx, batch_idx] = sum_sq
        end
        return
    end
    
    # Reduction kernel to sum across points for each batch
    function reduce_kernel!(out, partial_sums, n_points, n_batches)
        batch_idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
        
        if batch_idx ≤ n_batches
            sum_total = zero(eltype(partial_sums))
            for j in 1:n_points
                sum_total += partial_sums[j, batch_idx]
            end
            out[batch_idx] = sqrt(sum_total / n_points)
        end
        return
    end
    
    n_dims = size(P, 1)
    n_points = size(P, 2)
    n_batches = size(P, 3)
    
    # Allocate intermediate storage
    partial_sums = CUDA.zeros(eltype(P), n_points, n_batches)
    out = similar(P, n_batches)
    
    # 2D grid for computing squared differences
    threads_x = min(32, n_batches)
    threads_y = min(32, n_points)
    blocks_x = cld(n_batches, threads_x)
    blocks_y = cld(n_points, threads_y)
    
    kernel_2d! = CUDA.@cuda launch=false kernel_2d!(partial_sums, P, Q, n_dims, n_points, n_batches)
    kernel_2d!(partial_sums, P, Q, n_dims, n_points, n_batches; threads=(threads_x, threads_y), blocks=(blocks_x, blocks_y))
    
    # 1D grid for reduction
    reduce_kernel! = CUDA.@cuda launch=false reduce_kernel!(out, partial_sums, n_points, n_batches)
    threads_reduce = min(256, n_batches)
    blocks_reduce = cld(n_batches, threads_reduce)
    reduce_kernel!(out, partial_sums, n_points, n_batches; threads=threads_reduce, blocks=blocks_reduce)
    
    return out
end


end
