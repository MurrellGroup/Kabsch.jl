centroid(A::StaticArray; dims::Val=Val(2)) = mean(A; dims)

function kabsch(P::StaticMatrix{N,M,<:Number}, Q::StaticMatrix{N,M,<:Number}) where {N,M}
    reflection(U, V) = Diagonal(SVector{N}(ntuple(Returns(1), N-1)..., sign(det(U * V'))))
    size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    Pₜ, Qₜ = centroid(P), centroid(Q)
    P′, Q′ = P .- Pₜ, Q .- Qₜ
    H = P′ * Q′'
    U, Σ, V = svd(H)
    R = U * reflection(U, V) * V'
    return R, Pₜ, Qₜ
end
