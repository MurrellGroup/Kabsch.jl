module KabschStaticArraysExt

using Kabsch
using StaticArrays

using LinearAlgebra
using Statistics

Kabsch.centroid(A::StaticArray; dims::Val=Val(2)) = mean(A; dims)

function Kabsch.rmsd(P::StaticMatrix{N,M,<:Number}, Q::StaticMatrix{N,M,<:Number}) where {N,M}
    size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    return sqrt(mean(sum(abs2, P .- Q, dims=Val(1))))
end

function Kabsch.kabsch(P::StaticMatrix{N,M,<:Number}, Q::StaticMatrix{N,M,<:Number}) where {N,M}
    reflection(U, V) = Diagonal(SVector{N}(ntuple(Returns(1), N-1)..., sign(det(U * V'))))
    size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    Pₜ, Qₜ = centroid(P), centroid(Q)
    P′, Q′ = P .- Pₜ, Q .- Qₜ
    H = P′ * Q′'
    U, Σ, V = svd(H)
    R = U * reflection(U, V) * V'
    return R, Pₜ, Qₜ
end

function Kabsch.superimpose(Q::StaticMatrix{N,M,<:Number}, P::StaticMatrix{N,M,<:Number}) where {N,M}
    R̂, Pₜ, Qₜ = kabsch(P, Q)
    Q̂ = R̂ * (Q .- Qₜ) .+ Pₜ
    return Q̂
end

end
