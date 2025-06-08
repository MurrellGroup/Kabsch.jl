# Kabsch.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Kabsch.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Kabsch.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Kabsch.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Kabsch.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Kabsch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Kabsch.jl)

A Julia implementation of the Kabsch algorithm for finding the optimal rotation between two sets of points.

## Features

- Standard Kabsch algorithm for point cloud alignment
- Batched operations for multiple point sets
- StaticArrays support for optimal performance with small arrays
- CUDA GPU acceleration for large point clouds (optional)

## Usage

```julia
using Kabsch

# Two sets of points (N×M matrices, N dimensions, M points)
P = rand(3, 10)  # Reference points
Q = rand(3, 10)  # Points to align

# Find optimal rotation and centroids
R, P_centroid, Q_centroid = kabsch(P, Q)

# Directly superimpose Q onto P
Q_aligned = superimposed(Q, P)
```

### Batched Operations

Process multiple inputs using a batch dimension, utilizing batched operations:

```julia
# Batched arrays (N×M×B, B batches)
Ps = rand(3, 10, 8)
Qs = rand(3, 10, 8)

Rs, P_centroids, Q_centroids = kabsch(Ps, Qs)
```

### Static Arrays

For small point sets, use StaticArrays for better performance:

```julia
using StaticArrays

P = @SMatrix rand(3, 5)
Q = @SMatrix rand(3, 5)

R, Pt, Qt = kabsch(P, Q)
```

### GPU Support

CUDA acceleration is available when CUDA.jl is loaded:

```julia
using CUDA

P_gpu = CUDA.rand(3, 50, 100)  # 100 batches on GPU
Q_gpu = CUDA.rand(3, 50, 100)

R_gpu, Pt_gpu, Qt_gpu = kabsch(P_gpu, Q_gpu)
```

## See also
- `rmsd` and `superimpose!` in [BioStructures.jl](https://github.com/BioJulia/BioStructures.jl)
