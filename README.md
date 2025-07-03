# Kabsch.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Kabsch.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Kabsch.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Kabsch.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Kabsch.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Kabsch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Kabsch.jl)

A Julia implementation of the Kabsch algorithm for finding the optimal rotation between paired sets of points.

## Usage

```julia
using Kabsch, Manifolds

# Generate realistic test data
P = randn(3, 10)
R_true = rand(Rotations(3))  # Random rotation matrix
Q = R_true * centered(P) .+ randn(3)  # Rotate and translate P

# Find optimal alignment
R, P_centroid, Q_centroid = kabsch(P, Q)
Q_aligned = superimposed(Q, P)
```

### Batched Operations

Process multiple alignments simultaneously:

```julia
Ps = randn(3, 50, 8)  # 8 reference point sets
Qs = randn(3, 50, 8)  # 8 point sets to align, uncorrelated for ease of showcase

Rs, P_centroids, Q_centroids = kabsch(Ps, Qs)
Q_aligned_all = superimposed(Qs, Ps)
```

### Extensions

**StaticArrays:** Optimal performance for small point sets
```julia
using StaticArrays
P = @SMatrix randn(3, 5)
Q = @SMatrix randn(3, 5)  # uncorrelated for ease of showcase
R, Pt, Qt = kabsch(P, Q)
```

**CUDA:** GPU acceleration for 3D case
```julia
using CUDA
P_gpu = CUDA.randn(3, 100, 1000)  # 100 batches on GPU
Q_gpu = CUDA.randn(3, 100, 1000)  # uncorrelated for ease of showcase
R_gpu, _, _ = kabsch(P_gpu, Q_gpu)
```

## See also
- [BioStructures.jl](https://github.com/BioJulia/BioStructures.jl) for `rmsd` and `superimpose!` on molecular structures (including residue alignment)
