module Kabsch

using LinearAlgebra
using Statistics
using NNlib: batched_transpose, ⊠

export rmsd, kabsch, centroid
export superimpose

function rmsd(P::AbstractMatrix{<:Number}, Q::AbstractMatrix{<:Number})
    size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    return sqrt(mean(sum(abs2, P .- Q, dims=1)))
end

centroid(A::AbstractArray{<:Number}; dims=2) = mean(A; dims)

function kabsch(P::AbstractMatrix{<:Number}, Q::AbstractMatrix{<:Number})
    N = size(P, 1); @assert size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    Pₜ, Qₜ = centroid(P), centroid(Q)
    P′, Q′ = P .- Pₜ, Q .- Qₜ
    H = P′ * Q′'
    U, _, V = svd(H)
    V[:, end] .*= sign(det(U * V'))
    R = U * V'
    return R, Pₜ, Qₜ
end

function superimpose(Q::AbstractArray{<:Number}, P::AbstractArray{<:Number})
    R̂, Pₜ, Qₜ = kabsch(P, Q)
    Q̂ = R̂ ⊠ (Q .- Qₜ) .+ Pₜ
    return Q̂
end

rmsd(::typeof(superimpose), P, Q) = rmsd(superimpose(P, Q), Q)

include("batched.jl")

end
