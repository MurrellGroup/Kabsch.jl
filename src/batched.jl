function rmsd(P::AbstractArray{<:Number,3}, Q::AbstractArray{<:Number,3})
    size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    return vec(sqrt.(mean(sum(abs2, P .- Q, dims=1), dims=2)))
end

function batched_svd(A::AbstractArray{<:Number,3})
    U = similar(A)
    V = similar(A)
    @inbounds for k in axes(A, 3)
        U[:,:,k], _, V[:,:,k] = @views svd(A[:,:,k])
    end
    return U, V
end

function batched_det(A::AbstractArray{<:Number,3})
    out = similar(A, size(A, 3))
    @inbounds for k in axes(A, 3)
        out[k] = @views det(A[:,:,k])
    end
    return out
end

function kabsch(P::AbstractArray{<:Number,3}, Q::AbstractArray{<:Number,3})
    N = size(P, 1); @assert size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    Pₜ, Qₜ = centroid(P), centroid(Q)
    P′, Q′ = P .- Pₜ, Q .- Qₜ
    H = P′ ⊠ batched_transpose(Q′)
    U, V = batched_svd(H)
    d = sign.(batched_det(U ⊠ batched_transpose(V)))
    V[:, end, :] .*= d'
    R = U ⊠ batched_transpose(V)
    return R, Pₜ, Qₜ
end
