using Kabsch
using Test

using LinearAlgebra
using StaticArrays
using NNlib
using Manifolds: Rotations

@testset "Kabsch.jl" begin

    @testset for n_dims in 1:4, n_points in n_dims:4

        @testset "standard" begin
            R = rand(Rotations(n_dims))
            Q = randn(n_dims, n_points)
            Qₜ = centroid(Q)
            Pₜ = randn(n_dims, 1)
            P = R * (Q .- Qₜ) .+ Pₜ
            R̂, P̂ₜ, Q̂ₜ = @inferred kabsch(P, Q)
            @test R̂ ≈ R
            @test P̂ₜ ≈ Pₜ
            @test Q̂ₜ ≈ Qₜ

            @test superimpose(Q, P) ≈ P
        end

        @testset "batched" begin
            batch_size = 8
            R = stack(rand(Rotations(n_dims), 8))
            Q = randn(n_dims, n_points, batch_size)
            Qₜ = centroid(Q)
            Pₜ = randn(n_dims, 1, batch_size)
            P = R ⊠ (Q .- Qₜ) .+ Pₜ
            R̂, P̂ₜ, Q̂ₜ = @inferred kabsch(P, Q)
            @test R̂ ≈ R
            @test P̂ₜ ≈ Pₜ
            @test Q̂ₜ ≈ Qₜ

            @test superimpose(Q, P) ≈ P
        end

        @testset "static" begin
            R = rand(Rotations(n_dims)) |> SMatrix{n_dims,n_dims}
            Q = @SMatrix randn(n_dims, n_points)
            Qₜ = centroid(Q)
            Pₜ = @SMatrix randn(n_dims, 1)
            P = R * (Q .- Qₜ) .+ Pₜ
            R̂, P̂ₜ, Q̂ₜ = @inferred kabsch(P, Q)
            @test R̂ ≈ R
            @test P̂ₜ ≈ Pₜ
            @test Q̂ₜ ≈ Qₜ

            @test superimpose(Q, P) ≈ P
        end

    end

end
