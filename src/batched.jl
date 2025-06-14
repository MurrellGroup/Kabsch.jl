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

function kabsch_rotation(P::AbstractArray{<:Number,3}, Q::AbstractArray{<:Number,3})
    H = P ⊠ batched_transpose(Q)
    U, V = batched_svd(H)
    V[:, end, :] .*= sign.(batched_det(U ⊠ batched_transpose(V))')
    R = U ⊠ batched_transpose(V)
    return R
end

function superimposed(Q::AbstractArray{<:Number}, P::AbstractArray{<:Number})
    Pₜ, Qₜ = centroid(P), centroid(Q)
    Q_centered = centered(Q, Qₜ)
    R = kabsch_rotation(centered(P, Pₜ), Q_centered)
    return R ⊠ Q_centered .+ Pₜ
end
