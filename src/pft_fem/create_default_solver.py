#!/usr/bin/env python3
"""
Create precomputed default FEM solver for fast initialization.

This script generates a precomputed solver with default parameters:
- MNI152 space with posterior fossa restriction
- Bundled SUIT atlas regions
- Bundled MNI152 tissue segmentation
- Bundled HCP1065 DTI fiber orientations
- Fixed boundary conditions

The precomputed solver can be loaded with TumorGrowthSolver.load_default()
for ~100x faster initialization compared to building from scratch.

Usage:
    python -m pft_fem.create_default_solver
"""

import sys
import time
from pathlib import Path


def create_default_solver(output_dir: Path = None) -> None:
    """
    Create and save the default precomputed solver.

    Args:
        output_dir: Directory to save solver. Defaults to data/solvers/default_posterior_fossa.
    """
    print("Creating precomputed default FEM solver...")
    print("=" * 60)

    # Import here to show progress
    print("Loading modules...")
    start = time.time()

    from .biophysical_constraints import BiophysicalConstraints
    from .mesh import MeshGenerator
    from .fem import TumorGrowthSolver

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

    # Get posterior fossa mask from biophysical constraints
    mask = bc.compute_posterior_fossa_mask()
    print(f"  Posterior fossa mask shape: {mask.shape}")
    print(f"  Mask voxel count: {mask.sum()}")

    # Create mesh from mask
    mesh_gen = MeshGenerator()
    mesh = mesh_gen.from_mask(mask, voxel_size=(1.0, 1.0, 1.0))

    print(f"  Mesh nodes: {len(mesh.nodes)}")
    print(f"  Mesh elements: {len(mesh.elements)}")
    print(f"  Boundary nodes: {len(mesh.boundary_nodes)}")
    print(f"  Mesh created in {time.time() - start:.2f}s")

    # Step 3: Build FEM solver with biophysical constraints
    print("\nStep 3: Building FEM solver with tissue-specific properties...")
    start = time.time()

    solver = TumorGrowthSolver(
        mesh=mesh,
        boundary_condition="fixed",
        biophysical_constraints=bc,
    )

    print(f"  Mass matrix: {solver._mass_matrix.shape}, nnz={solver._mass_matrix.nnz}")
    print(f"  Stiffness matrix: {solver._stiffness_matrix.shape}, nnz={solver._stiffness_matrix.nnz}")
    print(f"  Diffusion matrix: {solver._diffusion_matrix.shape}, nnz={solver._diffusion_matrix.nnz}")
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
- **Boundary condition**: Fixed (skull immovable)
- **Voxel size**: 1.0 mm isotropic

## Usage

```python
from pft_fem import TumorGrowthSolver, TumorState

# Load precomputed solver (~100ms vs ~10s from scratch)
solver = TumorGrowthSolver.load_default()

# Create initial tumor state
state = TumorState.initial(
    num_nodes=len(solver.mesh.nodes),
    num_elements=len(solver.mesh.elements),
    seed_location=[1.0, -61.0, -34.0],  # Vermis in MNI coordinates
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

    args = parser.parse_args()

    try:
        create_default_solver(args.output_dir)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
