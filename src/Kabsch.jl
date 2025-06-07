module Kabsch

using LinearAlgebra
using StaticArrays
using Statistics
using NNlib: batched_transpose, ⊠

export kabsch, centroid
export superimpose

centroid(A::AbstractArray{<:Number}; dims=2) = mean(A; dims)

function kabsch(P::AbstractMatrix{<:Number}, Q::AbstractMatrix{<:Number})
    N = size(P, 1); @assert size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    Pₜ, Qₜ = centroid(P), centroid(Q)
    P′, Q′ = P .- Pₜ, Q .- Qₜ
    H = P′ * Q′'
    U, _, V = svd(H)
    d = sign(det(U * V'))
    V[:, end] .*= d
    R = U * V'
    return R, Pₜ, Qₜ
end

function superimpose(Q::AbstractArray{<:Number}, P::AbstractArray{<:Number})
    R̂, Pₜ, Qₜ = kabsch(P, Q)
    Q̂ = R̂ ⊠ (Q .- Qₜ) .+ Pₜ
    return Q̂
end

include("static.jl")
include("batched.jl")

end
