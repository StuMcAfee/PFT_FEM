#!/usr/bin/env python3
"""
Create precomputed default FEM solver for fast initialization.

This script generates a precomputed solver with default parameters:
- MNI152 space with posterior fossa restriction
- Bundled SUIT atlas regions
- Bundled MNI152 tissue segmentation
- Bundled HCP1065 DTI fiber orientations
- Fixed boundary conditions
- Coarse mesh (3mm voxels) for fast simulation with high-res output interpolation

The precomputed solver can be loaded with TumorGrowthSolver.load_default()
for ~100x faster initialization compared to building from scratch.

Usage:
    python -m pft_fem.create_default_solver
    python -m pft_fem.create_default_solver --mesh-voxel-size 2.0  # finer mesh
"""

import sys
import time
from pathlib import Path


def create_default_solver(
    output_dir: Path = None,
    mesh_voxel_size: float = 3.0,
) -> None:
    """
    Create and save the default precomputed solver.

    Args:
        output_dir: Directory to save solver. Defaults to data/solvers/default_posterior_fossa.
        mesh_voxel_size: Mesh voxel size in mm. Default 3.0mm for fast simulation.
                        Smaller values = more accurate but slower.
    """
    print("Creating precomputed default FEM solver...")
    print(f"  Mesh voxel size: {mesh_voxel_size}mm")
    print("=" * 60)

    # Import here to show progress
    print("Loading modules...")
    start = time.time()

    import numpy as np
    from scipy import ndimage

    from .biophysical_constraints import BiophysicalConstraints
    from .mesh import MeshGenerator
    from .fem import TumorGrowthSolver, SolverConfig

    print(f"  Modules loaded in {time.time() - start:.2f}s")

    # Determine output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "solvers" / "default_posterior_fossa"
    output_dir = Path(output_dir)

    # Step 1: Load biophysical constraints
    print("\nStep 1: Loading biophysical constraints...")
    start = time.time()

    bc = BiophysicalConstraints(
        posterior_fossa_only=True,
        use_suit_space=False,  # Use MNI152 space
    )
    bc.load_all_constraints()

    print(f"  Tissue segmentation loaded: {bc._segmentation.labels.shape}")
    print(f"  Fiber orientations loaded: {bc._fibers.vectors.shape}")
    print(f"  Loaded in {time.time() - start:.2f}s")

    # Step 2: Create posterior fossa mask and mesh
    print("\nStep 2: Creating mesh from posterior fossa region...")
    start = time.time()

    # Get posterior fossa bounding box mask
    pf_bounds_mask = bc.compute_posterior_fossa_mask()
    print(f"  Posterior fossa bounds shape: {pf_bounds_mask.shape}")
    print(f"  Posterior fossa bounds voxels: {pf_bounds_mask.sum()}")

    # Get actual brain tissue from segmentation (GM + WM, excluding CSF/background)
    tissue_labels = bc._segmentation.labels
    brain_tissue_mask = (tissue_labels == 2) | (tissue_labels == 3)  # GM=2, WM=3
    print(f"  Brain tissue voxels (GM+WM): {brain_tissue_mask.sum()}")

    # Combine: only brain tissue within posterior fossa bounds
    mask = pf_bounds_mask & brain_tissue_mask
    print(f"  Combined mask voxels (posterior fossa tissue): {mask.sum()}")

    # Get the affine from the segmentation to transform mesh to MNI physical coordinates
    # The segmentation affine maps voxel indices to MNI physical space
    original_affine = bc._segmentation.affine
    if original_affine is None:
        # Fallback: create standard MNI152 affine (center at AC)
        # Standard MNI152 1mm: origin at corner, center voxel (91, 109, 91) = MNI (0, 0, 0)
        original_affine = np.eye(4)
        original_affine[0, 3] = -90  # X offset to center
        original_affine[1, 3] = -126  # Y offset
        original_affine[2, 3] = -72   # Z offset

    print(f"  Original affine (MNI space):")
    print(f"    Voxel [0,0,0] -> MNI {original_affine[:3, 3]}")

    # Downsample mask if using coarse mesh (voxel_size > 1.5mm)
    original_voxel_size = 1.0  # MNI152 is 1mm isotropic
    coarse_factor = mesh_voxel_size / original_voxel_size

    if coarse_factor > 1.5:
        print(f"  Downsampling mask by factor {coarse_factor:.1f} for coarse mesh...")
        zoom_factors = tuple(1.0 / coarse_factor for _ in range(3))
        mask_coarse = ndimage.zoom(
            mask.astype(np.float32),
            zoom_factors,
            order=0  # Nearest neighbor for binary mask
        ) > 0.5
        coarse_voxel_size = (mesh_voxel_size, mesh_voxel_size, mesh_voxel_size)

        # Use the original affine without modification
        # The mesh generator scales nodes by voxel_size first (0-180 for 3mm voxels),
        # then the affine transforms to physical coordinates
        # This works correctly because: physical = affine @ (voxel_index * voxel_size)
        coarse_affine = original_affine.copy()

        print(f"  Coarse mask shape: {mask_coarse.shape}")
        print(f"  Coarse mask voxels: {mask_coarse.sum()}")
    else:
        mask_coarse = mask
        coarse_voxel_size = (original_voxel_size, original_voxel_size, original_voxel_size)
        coarse_affine = original_affine

    # Create mesh from mask at specified resolution
    # Pass the affine to transform mesh nodes to MNI physical coordinates
    mesh_gen = MeshGenerator()
    mesh = mesh_gen.from_mask(mask_coarse, voxel_size=coarse_voxel_size, affine=coarse_affine)

    print(f"  Mesh nodes: {len(mesh.nodes)}")
    print(f"  Mesh elements: {len(mesh.elements)}")
    print(f"  Boundary nodes: {len(mesh.boundary_nodes)}")
    print(f"  Mesh created in {time.time() - start:.2f}s")

    # Step 3: Build FEM solver with biophysical constraints and default config
    print("\nStep 3: Building FEM solver with tissue-specific properties...")
    start = time.time()

    # Use default solver config with coarse mesh settings
    solver_config = SolverConfig.default()
    solver_config = SolverConfig(
        mechanical_tol=solver_config.mechanical_tol,
        mechanical_maxiter=solver_config.mechanical_maxiter,
        use_amg=solver_config.use_amg,
        amg_cycle=solver_config.amg_cycle,
        amg_strength=solver_config.amg_strength,
        mesh_voxel_size=mesh_voxel_size,
        output_at_full_resolution=True,
    )

    solver = TumorGrowthSolver(
        mesh=mesh,
        boundary_condition="skull_csf_free",  # Fixed skull, free CSF (fourth ventricle)
        biophysical_constraints=bc,
        solver_config=solver_config,
    )

    print(f"  Mass matrix: {solver._mass_matrix.shape}, nnz={solver._mass_matrix.nnz}")
    print(f"  Stiffness matrix: {solver._stiffness_matrix.shape}, nnz={solver._stiffness_matrix.nnz}")
    print(f"  Diffusion matrix: {solver._diffusion_matrix.shape}, nnz={solver._diffusion_matrix.nnz}")
    print(f"  Solver config: mesh_voxel_size={mesh_voxel_size}mm, use_amg={solver_config.use_amg}")
    print(f"  Solver built in {time.time() - start:.2f}s")

    # Step 4: Save precomputed solver
    print(f"\nStep 4: Saving precomputed solver to {output_dir}...")
    start = time.time()

    solver.save(str(output_dir))

    # Calculate saved data size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Saved in {time.time() - start:.2f}s")

    # Create README
    readme_content = """# Precomputed Default FEM Solver

This directory contains a precomputed FEM solver for posterior fossa tumor simulations.

## Parameters

- **Coordinate space**: MNI152
- **Region**: Posterior fossa (cerebellum + brainstem)
- **Tissue segmentation**: MNI152 FAST segmentation (GM/WM/CSF)
- **Fiber orientations**: HCP1065 DTI atlas
- **Boundary condition**: Skull fixed, CSF free (fourth ventricle can be displaced)
- **Voxel size**: 1.0 mm isotropic

## Usage

```python
from pft_fem import TumorGrowthSolver, TumorState

# Load precomputed solver (~100ms vs ~10s from scratch)
solver = TumorGrowthSolver.load_default()

# Create initial tumor state
state = TumorState.initial(
    mesh=solver.mesh,
    seed_center=[2.0, -64.0, -36.0],  # Vermis/fourth ventricle in MNI coordinates
)

# Run simulation
for _ in range(100):
    state = solver.step(state, dt=0.1)
```

## Files

- `mesh.vtu` - Tetrahedral mesh (meshio format)
- `boundary_nodes.npy` - Fixed boundary node indices
- `solver_metadata.json` - Solver configuration and parameters
- `matrices/` - Precomputed sparse system matrices
  - `mass_matrix.npz` - Mass matrix for time integration
  - `stiffness_matrix.npz` - Mechanical stiffness matrix
  - `diffusion_matrix.npz` - Tumor diffusion matrix
- `precomputed/` - Precomputed element and node data
  - `element_volumes.npy` - Volume of each tetrahedron
  - `shape_gradients.pkl` - Shape function gradients
  - `node_tissues.npy` - Tissue type at each node
  - `node_fiber_directions.npy` - Fiber direction at each node
  - `element_properties.json` - Material properties per element

## Regenerating

To regenerate this precomputed solver:

```bash
python -m pft_fem.create_default_solver
```
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    print("\n" + "=" * 60)
    print("Precomputed default solver created successfully!")
    print(f"Load with: TumorGrowthSolver.load_default()")


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create precomputed default FEM solver for fast initialization."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/solvers/default_posterior_fossa)",
    )
    parser.add_argument(
        "--mesh-voxel-size",
        type=float,
        default=3.0,
        help="Mesh voxel size in mm (default: 3.0 for fast coarse simulation)",
    )

    args = parser.parse_args()

    try:
        create_default_solver(
            output_dir=args.output_dir,
            mesh_voxel_size=args.mesh_voxel_size,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
