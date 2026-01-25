# Precomputed Default FEM Solver

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
