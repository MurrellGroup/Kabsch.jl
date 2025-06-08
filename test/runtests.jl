using Kabsch
using Test

using LinearAlgebra
using NNlib
using Manifolds: Rotations

using StaticArrays

# ENV["KABSCH_TEST_CUDA"] = "true"
const KABSCH_TEST_CUDA = get(ENV, "KABSCH_TEST_CUDA", "false") == "true"
KABSCH_TEST_CUDA && begin using Pkg; Pkg.add("CUDA") end

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

            @test (@inferred superimposed(Q, P)) ≈ P

            @test rmsd(P, P) == 0
            @test rmsd(P, Q) != 0
            @test isapprox(rmsd(superimposed, P, Q), 0; atol=1e-10)
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

            @test (@inferred superimposed(Q, P)) ≈ P

            @test all(==(0), rmsd(P, P))
            @test all(!=(0), rmsd(P, Q))
            @test all(x -> isapprox(x, 0; atol=1e-10), rmsd(superimposed, P, Q))
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

            @test (@inferred superimposed(Q, P)) ≈ P
            @test superimposed(Q, P) isa typeof(Q)

            @test rmsd(P, P) == 0
            @test rmsd(P, Q) != 0
            @test isapprox(rmsd(superimposed, P, Q), 0; atol=1e-10)
        end

    end

    KABSCH_TEST_CUDA && include("ext_cuda/runtests.jl")

end
