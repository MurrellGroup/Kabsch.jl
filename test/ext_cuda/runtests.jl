using CUDA
using Kabsch
using Test
using LinearAlgebra
using NNlib
using Manifolds: Rotations

@testset "CUDA.jl" begin
    
    if CUDA.functional()
        
        @testset "batched_svd" begin
            n = 5
            A_cpu = randn(Float32, 3, 3, n)
            A_gpu = CuArray(A_cpu)
            
            U_cpu, V_cpu = Kabsch.batched_svd(A_cpu)
            U_gpu, V_gpu = Kabsch.batched_svd(A_gpu)
            
            # Check that both produce valid SVD decompositions
            # A ≈ U * S * V' (where S is diagonal with singular values)
            # Since we only return U and V, we verify U*V' gives a valid rotation-like matrix
            for i in 1:n
                # Verify the SVD reconstruction (up to singular values)
                R_cpu = V_cpu[:,:,i] * U_cpu[:,:,i]'
                R_gpu = Array(V_gpu[:,:,i] * U_gpu[:,:,i]')
                
                # Both should produce valid rotation matrices (orthogonal, det ≈ ±1)
                @test R_cpu' * R_cpu ≈ I atol=1e-5
                @test R_gpu' * R_gpu ≈ I atol=1e-5
                @test abs(det(R_cpu)) ≈ 1 atol=1e-5
                @test abs(det(R_gpu)) ≈ 1 atol=1e-5
            end
        end
        
        @testset "batched_det" begin
            n = 5
            A_cpu = randn(Float32, 3, 3, n)
            A_gpu = CuArray(A_cpu)
            
            det_cpu = Kabsch.batched_det(A_cpu)
            det_gpu = Kabsch.batched_det(A_gpu)
            
            @test det_cpu ≈ Array(det_gpu) atol=1e-5
            
            # Test specific determinants
            for i in 1:n
                @test det_cpu[i] ≈ det(A_cpu[:,:,i]) atol=1e-5
            end
        end
        
        @testset "kabsch on GPU" begin
            @testset for n_dims in 3:3, n_points in n_dims:4
                batch_size = 8
                
                # Generate test data on CPU
                R = stack(rand(Rotations(n_dims), batch_size))
                Q = randn(Float32, n_dims, n_points, batch_size)
                Qₜ = centroid(Q)
                Pₜ = randn(Float32, n_dims, 1, batch_size)
                P = R ⊠ (Q .- Qₜ) .+ Pₜ
                
                # Transfer to GPU
                P_gpu = CuArray(P)
                Q_gpu = CuArray(Q)
                
                # Run kabsch on GPU
                R̂_gpu, P̂ₜ_gpu, Q̂ₜ_gpu = @inferred kabsch(P_gpu, Q_gpu)
                
                # Transfer back to CPU for comparison
                R̂ = Array(R̂_gpu)
                P̂ₜ = Array(P̂ₜ_gpu)
                Q̂ₜ = Array(Q̂ₜ_gpu)
                
                @test R̂ ≈ R atol=1e-4
                @test P̂ₜ ≈ Pₜ atol=1e-4
                @test Q̂ₜ ≈ Qₜ atol=1e-4
            end
        end
        
        @testset "superimposed on GPU" begin
            @testset for n_dims in 3:3, n_points in n_dims:4
                batch_size = 8
                
                # Generate test data on CPU
                R = stack(rand(Rotations(n_dims), batch_size))
                Q = randn(Float32, n_dims, n_points, batch_size)
                Qₜ = centroid(Q)
                Pₜ = randn(Float32, n_dims, 1, batch_size)
                P = R ⊠ (Q .- Qₜ) .+ Pₜ
                
                # Transfer to GPU
                P_gpu = CuArray(P)
                Q_gpu = CuArray(Q)
                
                # Run superimposed on GPU
                Q̂_gpu = @inferred superimposed(Q_gpu, P_gpu)
                
                # Transfer back to CPU for comparison
                Q̂ = Array(Q̂_gpu)
                
                @test Q̂ ≈ P atol=1e-4
            end
        end
        
        @testset "rmsd on GPU" begin
            @testset for n_dims in 3:3, n_points in n_dims:4
                batch_size = 8
                
                # Generate test data on CPU
                P = randn(Float32, n_dims, n_points, batch_size)
                Q = randn(Float32, n_dims, n_points, batch_size)
                
                # Transfer to GPU
                P_gpu = CuArray(P)
                Q_gpu = CuArray(Q)
                
                # Compute RMSD on CPU and GPU
                rmsd_cpu = rmsd(P, Q)
                rmsd_gpu = rmsd(P_gpu, Q_gpu)
                
                @test Array(rmsd_gpu) ≈ rmsd_cpu atol=1e-4
                
                # Test self-RMSD is zero
                @test all(==(0), Array(rmsd(P_gpu, P_gpu)))
                
                # Test rmsd with superimposed
                R = stack(rand(Rotations(n_dims), batch_size))
                Q = randn(Float32, n_dims, n_points, batch_size)
                Qₜ = centroid(Q)
                Pₜ = randn(Float32, n_dims, 1, batch_size)
                P = R ⊠ (Q .- Qₜ) .+ Pₜ
                
                P_gpu = CuArray(P)
                Q_gpu = CuArray(Q)
                
                rmsd_aligned = rmsd(superimposed, P_gpu, Q_gpu)
                @test all(x -> isapprox(x, 0; atol=1e-4), Array(rmsd_aligned))
            end
        end
        
    else
        @warn "Skipped CUDA tests due to CUDA not being functional."
    end
    
end