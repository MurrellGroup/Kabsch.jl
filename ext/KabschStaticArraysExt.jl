module KabschStaticArraysExt

using Kabsch
using StaticArrays

using LinearAlgebra
using Statistics

Kabsch.centroid(A::StaticArray) = mean(A; dims=Val(2))

function Kabsch.rmsd(P::StaticMatrix{N,M,<:Number}, Q::StaticMatrix{N,M,<:Number}) where {N,M}
    size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    return sqrt(mean(sum(abs2, P .- Q, dims=Val(1))))
end

function Kabsch.kabsch_rotation(P::StaticMatrix{N,M,<:Number}, Q::StaticMatrix{N,M,<:Number}) where {N,M}
    reflection(U, V) = Diagonal(SVector{N}(ntuple(Returns(1), N-1)..., sign(det(U * V'))))
    H = P * Q'
    U, _, V = svd(H)
    R = U * reflection(U, V) * V'
    return R
end

end
