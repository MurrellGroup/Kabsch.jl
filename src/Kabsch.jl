module Kabsch

using LinearAlgebra
using Statistics
using NNlib: batched_transpose, ⊠

export rmsd
export centroid
export centered
export kabsch
export superimposed

"""
    rmsd(P, Q)

Return the Root Mean Square Deviation between two paired sets of points.
Note that this method does not align the two sets by itself.

```jldoctest
julia> P = randn(3, 4);

julia> rmsd(P, P) == 0
true

julia> rmsd(P, P .+ 1) == √3
true
```
"""
function rmsd(P::AbstractMatrix{<:Number}, Q::AbstractMatrix{<:Number})
    size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    return sqrt(mean(sum(abs2, P .- Q, dims=1)))
end

"""
    centroid(P)

Return the centroid of a point set `P`, reducing the second array dimension with `Statistics.mean`,
allowing for batch dimensions.
"""
@inline centroid(P::AbstractArray{<:Number}) = mean(P; dims=2)

"""
    centered(P)

Return the point set `P` centered at the origin, allowing for batch dimensions.
"""
@inline centered(P::AbstractArray{<:Number}, Pₜ = centroid(P)) = P .- Pₜ

function kabsch_rotation(P::AbstractMatrix{<:Number}, Q::AbstractMatrix{<:Number})
    H = P * Q'
    U, _, V = svd(H)
    V[:, end] .*= sign(det(U * V'))
    R = U * V'
    return R
end

"""
    kabsch(P, Q)

Returns a rotation matrix R, and centroids of P and Q, where [`rmsd`](@ref) is minimized
between a centered P and R applied to a centered Q.

```jldoctest
julia> using Manifolds

julia> Q = randn(3, 4);

julia> P = rand(Rotations(3)) * centered(Q) .+ randn(3);

julia> R, Pₜ, Qₜ = kabsch(P, Q);

julia> (P .- Pₜ) ≈ R * (Q .- Qₜ)
true
```
"""
function kabsch(P::AbstractArray{<:Number}, Q::AbstractArray{<:Number})
    N = size(P, 1); @assert size(P) == size(Q) || throw(ArgumentError("P and Q must have the same size"))
    Pₜ, Qₜ = centroid(P), centroid(Q)
    R = kabsch_rotation(centered(P, Pₜ), centered(Q, Qₜ))
    return R, Pₜ, Qₜ
end

mul(A::AbstractMatrix, B::AbstractMatrix) = A * B
mul(A::AbstractArray, B::AbstractArray) = A ⊠ B

"""
    superimposed(Q, P)

Returns Q superimposed on P.

```jldoctest
julia> using Manifolds

julia> Q = randn(3, 4);

julia> P = rand(Rotations(3)) * centered(Q) .+ randn(3);

julia> superimposed(Q, P) ≈ P
true
```
"""
function superimposed(Q::AbstractArray{<:Number}, P::AbstractArray{<:Number})
    R̂, Pₜ, Qₜ = kabsch(P, Q)
    Q_superimposed = mul(R̂, centered(Q, Qₜ)) .+ Pₜ
    return Q_superimposed
end

"""
    rmsd(::typeof(superimposed), P, Q)

Return the superimposed Root Mean Square Deviation between P and Q.

```jldoctest
julia> using Manifolds

julia> Q = randn(3, 4);

julia> P = rand(Rotations(3)) * centered(Q) .+ randn(3);

julia> isapprox(rmsd(superimposed, Q, P), 0, atol=1e-10)
true
```
"""
rmsd(::typeof(superimposed), P, Q) = rmsd(superimposed(P, Q), Q)

include("batched.jl")

end
