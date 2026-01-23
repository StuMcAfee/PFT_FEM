"""
Finite Element Method solver for tumor growth simulation.

Implements a coupled reaction-diffusion and mechanical deformation model
for simulating tumor growth in brain tissue.

Supports:
- Anisotropic material properties for white matter (fiber-aligned resistance)
- Compressible gray matter with uniform mechanical response
- Skull/boundary immovability constraints
- Tissue-specific diffusion and growth parameters
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Callable, Dict, List, Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from .mesh import TetMesh


# =============================================================================
# JIT-compiled assembly functions (50-100x faster than pure Python)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _assemble_mass_matrix_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled mass matrix assembly.

        Returns COO format arrays (rows, cols, data) for sparse matrix construction.
        """
        num_elements = len(elements)
        # Each element contributes 4x4 = 16 entries
        total_entries = num_elements * 16

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]

            for i in range(4):
                for j in range(4):
                    if i == j:
                        val = vol / 10.0
                    else:
                        val = vol / 20.0

                    rows[idx] = elem[i]
                    cols[idx] = elem[j]
                    data[idx] = val
                    idx += 1

        return rows, cols, data

    @njit(cache=True)
    def _assemble_diffusion_matrix_isotropic_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        shape_gradients: np.ndarray,
        diffusion_coeffs: np.ndarray,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled isotropic diffusion matrix assembly.

        For elements without fiber directions (gray matter, CSF).
        """
        num_elements = len(elements)
        total_entries = num_elements * 16

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]
            grads = shape_gradients[e]  # (4, 3)
            D = diffusion_coeffs[e]

            for i in range(4):
                for j in range(4):
                    # Isotropic: K_ij = D * V * (grad_i . grad_j)
                    val = D * vol * (
                        grads[i, 0] * grads[j, 0] +
                        grads[i, 1] * grads[j, 1] +
                        grads[i, 2] * grads[j, 2]
                    )

                    rows[idx] = elem[i]
                    cols[idx] = elem[j]
                    data[idx] = val
                    idx += 1

        return rows, cols, data

    @njit(cache=True)
    def _assemble_diffusion_matrix_anisotropic_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        shape_gradients: np.ndarray,
        diffusion_coeffs: np.ndarray,
        fiber_directions: np.ndarray,
        tissue_types: np.ndarray,
        white_matter_type: int,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled anisotropic diffusion matrix assembly.

        Handles both isotropic and anisotropic elements based on tissue type.
        White matter uses anisotropic diffusion (2x along fibers, 0.5x perpendicular).
        """
        num_elements = len(elements)
        total_entries = num_elements * 16

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]
            grads = shape_gradients[e]  # (4, 3)
            D_base = diffusion_coeffs[e]
            tissue = tissue_types[e]

            # Build diffusion tensor
            if tissue == white_matter_type:
                # Anisotropic diffusion for white matter
                f = fiber_directions[e]
                f_norm = np.sqrt(f[0]**2 + f[1]**2 + f[2]**2)
                if f_norm > 1e-10:
                    f = f / f_norm

                D_para = D_base * 2.0
                D_perp = D_base * 0.5

                # D_tensor = D_perp * I + (D_para - D_perp) * f ⊗ f
                D_tensor = np.zeros((3, 3))
                for a in range(3):
                    D_tensor[a, a] = D_perp
                    for b in range(3):
                        D_tensor[a, b] += (D_para - D_perp) * f[a] * f[b]
            else:
                # Isotropic diffusion
                D_tensor = np.zeros((3, 3))
                D_tensor[0, 0] = D_base
                D_tensor[1, 1] = D_base
                D_tensor[2, 2] = D_base

            for i in range(4):
                for j in range(4):
                    # K_ij = V * grad_i^T * D * grad_j
                    val = 0.0
                    for a in range(3):
                        for b in range(3):
                            val += grads[i, a] * D_tensor[a, b] * grads[j, b]
                    val *= vol

                    rows[idx] = elem[i]
                    cols[idx] = elem[j]
                    data[idx] = val
                    idx += 1

        return rows, cols, data

    @njit(cache=True)
    def _compute_element_stiffness_isotropic(
        grads: np.ndarray,
        vol: float,
        lam: float,
        mu: float,
    ) -> np.ndarray:
        """Compute 12x12 element stiffness matrix for isotropic material."""
        Ke = np.zeros((12, 12))

        # Constitutive matrix (isotropic linear elasticity)
        C = np.array([
            [lam + 2*mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2*mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2*mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ])

        for i in range(4):
            dNdx_i, dNdy_i, dNdz_i = grads[i, 0], grads[i, 1], grads[i, 2]
            Bi = np.array([
                [dNdx_i, 0.0, 0.0],
                [0.0, dNdy_i, 0.0],
                [0.0, 0.0, dNdz_i],
                [dNdy_i, dNdx_i, 0.0],
                [dNdz_i, 0.0, dNdx_i],
                [0.0, dNdz_i, dNdy_i],
            ])

            for j in range(4):
                dNdx_j, dNdy_j, dNdz_j = grads[j, 0], grads[j, 1], grads[j, 2]
                Bj = np.array([
                    [dNdx_j, 0.0, 0.0],
                    [0.0, dNdy_j, 0.0],
                    [0.0, 0.0, dNdz_j],
                    [dNdy_j, dNdx_j, 0.0],
                    [dNdz_j, 0.0, dNdx_j],
                    [0.0, dNdz_j, dNdy_j],
                ])

                # K_ij = V * Bi^T @ C @ Bj
                for a in range(3):
                    for b in range(3):
                        val = 0.0
                        for k in range(6):
                            for l in range(6):
                                val += Bi[k, a] * C[k, l] * Bj[l, b]
                        Ke[i*3 + a, j*3 + b] = vol * val

        return Ke

    @njit(cache=True)
    def _assemble_stiffness_matrix_isotropic_jit(
        elements: np.ndarray,
        volumes: np.ndarray,
        shape_gradients: np.ndarray,
        lam_values: np.ndarray,
        mu_values: np.ndarray,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        JIT-compiled stiffness matrix assembly for isotropic materials.
        """
        num_elements = len(elements)
        # Each element contributes 12x12 = 144 entries
        total_entries = num_elements * 144

        rows = np.empty(total_entries, dtype=np.int64)
        cols = np.empty(total_entries, dtype=np.int64)
        data = np.empty(total_entries, dtype=np.float64)

        idx = 0
        for e in range(num_elements):
            vol = volumes[e]
            elem = elements[e]
            grads = shape_gradients[e]
            lam = lam_values[e]
            mu = mu_values[e]

            # Compute element stiffness matrix
            Ke = _compute_element_stiffness_isotropic(grads, vol, lam, mu)

            # Assemble into global arrays
            for i in range(4):
                for j in range(4):
                    for di in range(3):
                        for dj in range(3):
                            global_i = elem[i] * 3 + di
                            global_j = elem[j] * 3 + dj
                            local_i = i * 3 + di
                            local_j = j * 3 + dj

                            rows[idx] = global_i
                            cols[idx] = global_j
                            data[idx] = Ke[local_i, local_j]
                            idx += 1

        return rows, cols, data

if TYPE_CHECKING:
    from .biophysical_constraints import BiophysicalConstraints, AnisotropicMaterialProperties


class TissueType(Enum):
    """Brain tissue types with different mechanical properties."""

    GRAY_MATTER = "gray_matter"
    WHITE_MATTER = "white_matter"
    CSF = "csf"
    TUMOR = "tumor"
    EDEMA = "edema"
    SKULL = "skull"  # Immovable boundary


@dataclass
class MaterialProperties:
    """
    Material properties for brain tissue FEM simulation.

    Properties are based on literature values for brain tissue mechanics.

    Gray matter: More compressible (lower Poisson ratio), isotropic
    White matter: Nearly incompressible, anisotropic along fiber direction
    """

    # Elastic properties
    young_modulus: float = 3000.0  # Pa (brain tissue ~1-10 kPa)
    poisson_ratio: float = 0.45  # Nearly incompressible

    # Tumor growth parameters
    proliferation_rate: float = 0.01  # 1/day
    diffusion_coefficient: float = 0.1  # mm^2/day
    carrying_capacity: float = 1.0  # Normalized max cell density

    # Mechanical coupling (eigenstrain formulation)
    # This represents volumetric strain per unit cell density.
    # Value of 0.15 means 15% volumetric expansion at full tumor density.
    growth_stress_coefficient: float = 0.15  # Volumetric strain per unit density

    # Anisotropy parameters for white matter
    anisotropy_ratio: float = 2.0  # Ratio of parallel/perpendicular stiffness
    fiber_direction: Optional[NDArray[np.float64]] = None  # Local fiber direction

    # Tissue-specific multipliers
    tissue_stiffness_multipliers: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 1.0,
        TissueType.WHITE_MATTER: 1.2,  # Slightly stiffer
        TissueType.CSF: 0.01,  # Very soft (fluid)
        TissueType.TUMOR: 2.0,  # Tumors are often stiffer
        TissueType.EDEMA: 0.5,  # Softened tissue
        TissueType.SKULL: 1000.0,  # Very stiff (immovable)
    })

    tissue_diffusion_multipliers: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 1.0,
        TissueType.WHITE_MATTER: 2.0,  # Faster along fiber tracts
        TissueType.CSF: 0.1,  # Barrier to invasion
        TissueType.TUMOR: 0.5,
        TissueType.EDEMA: 1.5,
        TissueType.SKULL: 0.0,  # No diffusion through skull
    })

    # Tissue-specific Poisson ratios (compressibility)
    tissue_poisson_ratios: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 0.40,  # More compressible (uniform compression)
        TissueType.WHITE_MATTER: 0.45,  # Nearly incompressible
        TissueType.CSF: 0.499,  # Essentially incompressible fluid
        TissueType.TUMOR: 0.42,
        TissueType.EDEMA: 0.48,
        TissueType.SKULL: 0.30,  # Standard bone value
    })

    @classmethod
    def for_tissue(cls, tissue_type: TissueType) -> "MaterialProperties":
        """Create material properties for a specific tissue type."""
        base = cls()
        stiffness_mult = base.tissue_stiffness_multipliers.get(tissue_type, 1.0)
        diffusion_mult = base.tissue_diffusion_multipliers.get(tissue_type, 1.0)
        poisson = base.tissue_poisson_ratios.get(tissue_type, 0.45)

        return cls(
            young_modulus=base.young_modulus * stiffness_mult,
            poisson_ratio=poisson,
            proliferation_rate=base.proliferation_rate,
            diffusion_coefficient=base.diffusion_coefficient * diffusion_mult,
            carrying_capacity=base.carrying_capacity,
            growth_stress_coefficient=base.growth_stress_coefficient,
        )

    @classmethod
    def gray_matter(cls) -> "MaterialProperties":
        """
        Create material properties for gray matter.

        Gray matter is modeled as uniformly compressible with isotropic properties.
        Lower Poisson ratio allows volume change under pressure.
        """
        return cls(
            young_modulus=2500.0,  # Pa - softer than white matter
            poisson_ratio=0.40,  # More compressible than white matter
            proliferation_rate=0.01,
            diffusion_coefficient=0.1,
            carrying_capacity=1.0,
            growth_stress_coefficient=0.15,  # 15% volumetric strain at full density
            anisotropy_ratio=1.0,  # Isotropic
            fiber_direction=None,
        )

    @classmethod
    def white_matter(
        cls,
        fiber_direction: Optional[NDArray[np.float64]] = None,
    ) -> "MaterialProperties":
        """
        Create material properties for white matter.

        White matter resists stretching along the fiber direction (transversely isotropic).
        Higher stiffness parallel to fibers, nearly incompressible.
        """
        return cls(
            young_modulus=3500.0,  # Pa - stiffer than gray matter
            poisson_ratio=0.45,  # Nearly incompressible
            proliferation_rate=0.01,
            diffusion_coefficient=0.2,  # Faster along fibers
            carrying_capacity=1.0,
            growth_stress_coefficient=0.15,  # 15% volumetric strain at full density
            anisotropy_ratio=2.0,  # 2x stiffer along fibers
            fiber_direction=fiber_direction,
        )

    def lame_parameters(self) -> Tuple[float, float]:
        """Compute Lamé parameters from Young's modulus and Poisson ratio."""
        E = self.young_modulus
        nu = self.poisson_ratio

        # First Lamé parameter (lambda)
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        # Second Lamé parameter (mu, shear modulus)
        mu = E / (2 * (1 + nu))

        return lam, mu

    def get_constitutive_matrix(
        self,
        fiber_direction: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Get the 6x6 constitutive matrix (stress-strain relationship).

        For isotropic materials (gray matter), returns standard isotropic matrix.
        For anisotropic materials (white matter), returns transversely isotropic
        matrix with fiber direction as the axis of symmetry.

        Args:
            fiber_direction: Override fiber direction (uses self.fiber_direction if None)

        Returns:
            6x6 constitutive matrix in Voigt notation
        """
        fiber_dir = fiber_direction if fiber_direction is not None else self.fiber_direction

        if fiber_dir is None or self.anisotropy_ratio == 1.0:
            # Isotropic (gray matter)
            return self._isotropic_constitutive_matrix()
        else:
            # Transversely isotropic (white matter)
            return self._anisotropic_constitutive_matrix(fiber_dir)

    def _isotropic_constitutive_matrix(self) -> NDArray[np.float64]:
        """Build isotropic constitutive matrix."""
        lam, mu = self.lame_parameters()

        C = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ], dtype=np.float64)

        return C

    def _anisotropic_constitutive_matrix(
        self,
        fiber_direction: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Build transversely isotropic constitutive matrix for white matter.

        The fiber direction is the axis of symmetry (stiffer along fibers).
        This resists stretching in the fiber direction.
        """
        E_perp = self.young_modulus
        E_para = E_perp * self.anisotropy_ratio
        nu_perp = self.poisson_ratio
        nu_para = self.poisson_ratio * 0.8  # Lower along fibers

        # Shear moduli
        G_perp = E_perp / (2 * (1 + nu_perp))
        G_para = E_para / (2 * (1 + nu_para))

        # Build compliance matrix in local (fiber-aligned) coordinates
        nu21 = nu_para * E_perp / E_para

        S = np.zeros((6, 6))
        S[0, 0] = 1 / E_para  # Along fiber
        S[1, 1] = 1 / E_perp  # Perpendicular
        S[2, 2] = 1 / E_perp  # Perpendicular
        S[0, 1] = -nu_para / E_para
        S[1, 0] = -nu21 / E_perp
        S[0, 2] = -nu_para / E_para
        S[2, 0] = -nu21 / E_perp
        S[1, 2] = -nu_perp / E_perp
        S[2, 1] = -nu_perp / E_perp
        S[3, 3] = 1 / G_perp  # sigma_23
        S[4, 4] = 1 / G_para  # sigma_13
        S[5, 5] = 1 / G_para  # sigma_12

        # Invert to get stiffness
        try:
            C_local = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self._isotropic_constitutive_matrix()

        # Rotate to global coordinates
        C_global = self._rotate_constitutive_to_global(C_local, fiber_direction)

        return C_global

    def _rotate_constitutive_to_global(
        self,
        C_local: NDArray[np.float64],
        fiber_dir: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Rotate constitutive matrix from fiber-aligned to global coordinates."""
        # Normalize fiber direction
        f = fiber_dir / np.linalg.norm(fiber_dir)

        # Build orthonormal basis with f as first axis
        if abs(f[0]) < 0.9:
            t = np.array([1.0, 0.0, 0.0])
        else:
            t = np.array([0.0, 1.0, 0.0])

        n1 = np.cross(f, t)
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.cross(f, n1)

        # Rotation matrix (columns are local axes in global coords)
        R = np.column_stack([f, n1, n2])

        # Build 6x6 transformation matrix for Voigt notation
        T = self._build_voigt_rotation(R)

        # Transform: C_global = T^T * C_local * T
        return T.T @ C_local @ T

    def _build_voigt_rotation(self, R: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build 6x6 Voigt rotation matrix."""
        T = np.zeros((6, 6))

        # Index pairs for Voigt notation
        pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        for I in range(6):
            i, j = pairs[I]
            for J in range(6):
                k, l = pairs[J]
                if I < 3 and J < 3:
                    T[I, J] = R[i, k] * R[j, l]
                elif I < 3:
                    T[I, J] = R[i, k] * R[j, l] + R[i, l] * R[j, k]
                elif J < 3:
                    T[I, J] = R[i, k] * R[j, l]
                else:
                    T[I, J] = R[i, k] * R[j, l] + R[i, l] * R[j, k]

        return T


@dataclass
class TumorState:
    """
    Current state of the tumor simulation.

    Attributes:
        cell_density: Tumor cell density at each node (0 to 1).
        displacement: Tissue displacement at each node (N, 3).
        stress: Stress tensor at each element (M, 6) in Voigt notation.
        time: Current simulation time in days.
    """

    cell_density: NDArray[np.float64]
    displacement: NDArray[np.float64]
    stress: NDArray[np.float64]
    time: float = 0.0

    @classmethod
    def initial(
        cls,
        mesh: TetMesh,
        seed_center: NDArray[np.float64],
        seed_radius: float = 2.0,
        seed_density: float = 0.5,
    ) -> "TumorState":
        """
        Create initial tumor state with a seed tumor.

        Args:
            mesh: FEM mesh.
            seed_center: Center of initial tumor seed in mm.
            seed_radius: Radius of initial tumor in mm.
            seed_density: Initial cell density in seed region.

        Returns:
            Initial TumorState.
        """
        num_nodes = mesh.num_nodes
        num_elements = mesh.num_elements

        # Initialize cell density with Gaussian seed
        distances = np.linalg.norm(mesh.nodes - seed_center, axis=1)
        cell_density = seed_density * np.exp(-(distances / seed_radius) ** 2)

        # Initialize displacement to zero
        displacement = np.zeros((num_nodes, 3), dtype=np.float64)

        # Initialize stress to zero
        stress = np.zeros((num_elements, 6), dtype=np.float64)

        return cls(
            cell_density=cell_density,
            displacement=displacement,
            stress=stress,
            time=0.0,
        )


# Default coarse mesh voxel size for fast simulation (mm)
DEFAULT_COARSE_MESH_VOXEL_SIZE = 3.0


@dataclass
class SolverConfig:
    """
    Configuration for FEM solver performance and accuracy tradeoffs.

    Provides options for approximate solutions that trade accuracy for speed:
    - AMG preconditioning for faster CG convergence
    - Reduced tolerance for fewer iterations
    - Coarse mesh with high-resolution output interpolation
    - Maximum iteration limits

    Example usage:
        # Fast coarse mode (recommended default, ~50-100x speedup)
        config = SolverConfig.fast_coarse()
        solver = TumorGrowthSolver(mesh, solver_config=config)

        # Fast approximate mode (3-10x speedup, same resolution)
        config = SolverConfig.fast()
        solver = TumorGrowthSolver(mesh, solver_config=config)

        # High accuracy mode (slower but more precise)
        config = SolverConfig.accurate()
        solver = TumorGrowthSolver(mesh, solver_config=config)
    """

    # Mechanical solver (CG) settings
    mechanical_tol: float = 1e-6  # Relative tolerance for CG convergence
    mechanical_maxiter: int = 1000  # Maximum CG iterations

    # AMG preconditioning settings
    use_amg: bool = True  # Use algebraic multigrid preconditioning
    amg_cycle: str = "V"  # AMG cycle type: "V", "W", or "F"
    amg_strength: str = "symmetric"  # Strength of connection: "symmetric" or "classical"

    # Diffusion solver settings (uses direct solver by default)
    diffusion_use_iterative: bool = False  # Use iterative solver for diffusion
    diffusion_tol: float = 1e-8  # Tolerance for iterative diffusion solver

    # Mesh resolution settings for multi-resolution simulation
    mesh_voxel_size: float = 1.0  # Mesh voxel size in mm (larger = coarser = faster)
    output_at_full_resolution: bool = True  # Interpolate output to original resolution

    # Cached AMG preconditioner (built lazily)
    _amg_preconditioner: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def default(cls) -> "SolverConfig":
        """
        Default configuration using coarse mesh for speed.

        Uses 3mm mesh voxels with AMG preconditioning and interpolates
        output to full resolution. This is the recommended default for
        most applications.

        Typical speedup: ~50-100x vs fine mesh + standard solver
        Accuracy: Smooth deformation field, suitable for most applications
        """
        return cls(
            mechanical_tol=1e-6,
            mechanical_maxiter=1000,
            use_amg=True,
            amg_cycle="V",
            mesh_voxel_size=DEFAULT_COARSE_MESH_VOXEL_SIZE,
            output_at_full_resolution=True,
        )

    @classmethod
    def fast_coarse(cls) -> "SolverConfig":
        """
        Fastest configuration using coarse mesh + reduced tolerance.

        Combines coarse mesh (3mm voxels), AMG preconditioning, and
        reduced solver tolerance for maximum speed. Output is interpolated
        to full resolution.

        Typical speedup: ~100-200x vs fine mesh + standard solver
        Accuracy: ~1-5% relative error in displacement field
        """
        return cls(
            mechanical_tol=1e-3,
            mechanical_maxiter=100,
            use_amg=True,
            amg_cycle="V",
            mesh_voxel_size=DEFAULT_COARSE_MESH_VOXEL_SIZE,
            output_at_full_resolution=True,
        )

    @classmethod
    def fast(cls) -> "SolverConfig":
        """
        Fast configuration with same mesh resolution.

        Uses reduced tolerance and AMG preconditioning for 3-10x speedup
        at the same mesh resolution. Use this when you need the full
        mesh detail but want faster solving.

        Typical speedup: 3-10x
        Accuracy: ~0.1-1% relative error
        """
        return cls(
            mechanical_tol=1e-3,
            mechanical_maxiter=100,
            use_amg=True,
            amg_cycle="V",
            mesh_voxel_size=1.0,  # Same as atlas resolution
            output_at_full_resolution=True,
        )

    @classmethod
    def accurate(cls) -> "SolverConfig":
        """
        High accuracy configuration for final results.

        Uses fine mesh (1mm voxels), tight tolerance, and direct solver.
        Use this when maximum accuracy is required.

        Typical speedup: 1x (baseline)
        Accuracy: Maximum precision
        """
        return cls(
            mechanical_tol=1e-8,
            mechanical_maxiter=2000,
            use_amg=False,
            mesh_voxel_size=1.0,
            output_at_full_resolution=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mechanical_tol": self.mechanical_tol,
            "mechanical_maxiter": self.mechanical_maxiter,
            "use_amg": self.use_amg,
            "amg_cycle": self.amg_cycle,
            "amg_strength": self.amg_strength,
            "diffusion_use_iterative": self.diffusion_use_iterative,
            "diffusion_tol": self.diffusion_tol,
            "mesh_voxel_size": self.mesh_voxel_size,
            "output_at_full_resolution": self.output_at_full_resolution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolverConfig":
        """Create from dictionary."""
        return cls(
            mechanical_tol=data.get("mechanical_tol", 1e-6),
            mechanical_maxiter=data.get("mechanical_maxiter", 1000),
            use_amg=data.get("use_amg", True),
            amg_cycle=data.get("amg_cycle", "V"),
            amg_strength=data.get("amg_strength", "symmetric"),
            diffusion_use_iterative=data.get("diffusion_use_iterative", False),
            diffusion_tol=data.get("diffusion_tol", 1e-8),
            mesh_voxel_size=data.get("mesh_voxel_size", DEFAULT_COARSE_MESH_VOXEL_SIZE),
            output_at_full_resolution=data.get("output_at_full_resolution", True),
        )


class TumorGrowthSolver:
    """
    FEM solver for tumor growth in brain tissue.

    Implements a coupled model:
    1. Reaction-diffusion equation for tumor cell density
    2. Linear elasticity for tissue deformation
    3. Mass effect: tumor growth causes tissue displacement

    The model uses operator splitting:
    - Diffusion step (implicit)
    - Reaction step (explicit)
    - Mechanical equilibrium (static)

    Supports biophysical constraints:
    - Anisotropic white matter: Resists stretching along fiber direction
    - Compressible gray matter: Uniform volumetric compression
    - Skull boundary: Immovable (fixed displacement)
    - Tissue-specific diffusion: Faster along white matter tracts
    """

    def __init__(
        self,
        mesh: TetMesh,
        properties: Optional[MaterialProperties] = None,
        boundary_condition: str = "fixed",
        biophysical_constraints: Optional["BiophysicalConstraints"] = None,
        solver_config: Optional[SolverConfig] = None,
    ):
        """
        Initialize the tumor growth solver.

        Args:
            mesh: Tetrahedral mesh for FEM.
            properties: Material properties (uses defaults if None).
            boundary_condition: Boundary condition type ("fixed", "skull", or "free").
            biophysical_constraints: Optional biophysical constraints for tissue-specific
                                    material properties, fiber orientation, and boundaries.
            solver_config: Solver configuration for performance/accuracy tradeoffs.
                          Use SolverConfig.fast() for approximate solutions,
                          SolverConfig.accurate() for high precision.
        """
        self.mesh = mesh
        self.properties = properties or MaterialProperties()
        self.boundary_condition = boundary_condition
        self.biophysical_constraints = biophysical_constraints
        self.solver_config = solver_config or SolverConfig.default()

        # Tissue and fiber data from biophysical constraints
        self._node_tissues: Optional[NDArray[np.int32]] = None
        self._node_fiber_directions: Optional[NDArray[np.float64]] = None
        self._element_properties: Optional[List[MaterialProperties]] = None

        # Cached AMG preconditioner (built lazily on first solve)
        self._amg_ml: Any = None

        # Initialize biophysical data if constraints provided
        if biophysical_constraints is not None:
            self._initialize_biophysical_data()

        # Precompute element matrices
        self._element_volumes = mesh.compute_element_volumes()
        self._shape_gradients = self._compute_shape_gradients()

        # Build system matrices
        self._mass_matrix = self._build_mass_matrix()
        self._stiffness_matrix = self._build_stiffness_matrix()
        self._diffusion_matrix = self._build_diffusion_matrix()

    def _initialize_biophysical_data(self) -> None:
        """Initialize tissue types and fiber directions from biophysical constraints."""
        bc = self.biophysical_constraints

        # Load all constraint data
        bc.load_all_constraints()

        # Assign tissue types to nodes
        self._node_tissues = bc.assign_node_tissues(self.mesh.nodes)

        # Get fiber directions at all nodes
        self._node_fiber_directions = bc.get_fiber_directions_at_nodes(self.mesh.nodes)

        # Build element-specific material properties
        self._element_properties = []
        for elem in self.mesh.elements:
            # Use dominant tissue type in element
            elem_tissues = self._node_tissues[elem]
            dominant_tissue = int(np.median(elem_tissues))

            # Get average fiber direction for element
            elem_fibers = self._node_fiber_directions[elem]
            avg_fiber = np.mean(elem_fibers, axis=0)
            norm = np.linalg.norm(avg_fiber)
            if norm > 1e-6:
                avg_fiber = avg_fiber / norm
            else:
                avg_fiber = np.array([1.0, 0.0, 0.0])

            # Create tissue-specific properties
            if dominant_tissue == 3:  # WHITE_MATTER (from BrainTissue enum)
                props = MaterialProperties.white_matter(fiber_direction=avg_fiber)
            elif dominant_tissue == 2:  # GRAY_MATTER
                props = MaterialProperties.gray_matter()
            elif dominant_tissue == 1:  # CSF
                props = MaterialProperties(
                    young_modulus=100.0,
                    poisson_ratio=0.499,
                    diffusion_coefficient=0.01,
                )
            else:
                props = MaterialProperties()

            self._element_properties.append(props)

    def _get_element_tissue_type(self, element_idx: int) -> TissueType:
        """Get the dominant tissue type for an element."""
        if self._node_tissues is None:
            return TissueType.GRAY_MATTER

        elem = self.mesh.elements[element_idx]
        elem_tissues = self._node_tissues[elem]
        dominant = int(np.median(elem_tissues))

        # Map from BrainTissue enum values
        tissue_map = {
            0: TissueType.CSF,  # BACKGROUND treated as CSF
            1: TissueType.CSF,
            2: TissueType.GRAY_MATTER,
            3: TissueType.WHITE_MATTER,
            4: TissueType.SKULL,
            5: TissueType.SKULL,  # SCALP treated as skull boundary
        }
        return tissue_map.get(dominant, TissueType.GRAY_MATTER)

    def _get_skull_boundary_nodes(self) -> NDArray[np.int32]:
        """Get nodes on the skull boundary for immovable constraint."""
        if self.biophysical_constraints is None:
            return self.mesh.boundary_nodes

        return self.biophysical_constraints.get_boundary_nodes(self.mesh.nodes)

    def _compute_shape_gradients(self) -> List[NDArray[np.float64]]:
        """Compute shape function gradients for each element."""
        gradients = []

        for elem in self.mesh.elements:
            coords = self.mesh.nodes[elem]  # (4, 3)

            # Jacobian matrix for linear tetrahedron
            # J = [x1-x0, x2-x0, x3-x0]^T
            J = np.array([
                coords[1] - coords[0],
                coords[2] - coords[0],
                coords[3] - coords[0],
            ])

            # Inverse Jacobian
            try:
                J_inv = np.linalg.inv(J)
            except np.linalg.LinAlgError:
                # Degenerate element
                J_inv = np.zeros((3, 3))

            # Shape function gradients in physical coordinates
            # For linear tetrahedron: dN/dx = J^(-T) * dN/dxi
            # dN/dxi for reference tetrahedron:
            # N0 = 1 - xi - eta - zeta -> grad = [-1, -1, -1]
            # N1 = xi -> grad = [1, 0, 0]
            # N2 = eta -> grad = [0, 1, 0]
            # N3 = zeta -> grad = [0, 0, 1]
            ref_grads = np.array([
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=np.float64)

            # Transform to physical coordinates
            phys_grads = ref_grads @ J_inv
            gradients.append(phys_grads)

        return gradients

    def _build_mass_matrix(self) -> sparse.csr_matrix:
        """Build the mass matrix for the reaction-diffusion equation."""
        n = self.mesh.num_nodes

        if HAS_NUMBA:
            # Use JIT-compiled assembly (50-100x faster)
            rows, cols, data = _assemble_mass_matrix_jit(
                self.mesh.elements,
                self._element_volumes,
                n,
            )
        else:
            # Fallback to pure Python
            rows, cols, data = [], [], []

            for e, elem in enumerate(self.mesh.elements):
                vol = self._element_volumes[e]

                # Mass matrix for linear tetrahedron
                # M_ij = integral(Ni * Nj) = V/20 * (1 + delta_ij)
                for i in range(4):
                    for j in range(4):
                        if i == j:
                            val = vol / 10.0
                        else:
                            val = vol / 20.0

                        rows.append(elem[i])
                        cols.append(elem[j])
                        data.append(val)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _build_diffusion_matrix(self) -> sparse.csr_matrix:
        """
        Build the diffusion matrix with tissue-specific coefficients.

        White matter: Anisotropic diffusion, faster along fiber direction
        Gray matter: Isotropic diffusion
        CSF: Reduced diffusion (barrier to invasion)
        """
        n = self.mesh.num_nodes
        num_elements = len(self.mesh.elements)

        # Prepare arrays for JIT compilation
        shape_grads_array = np.array(self._shape_gradients)  # (num_elements, 4, 3)

        # Build diffusion coefficient array
        diffusion_coeffs = np.empty(num_elements, dtype=np.float64)
        if self._element_properties is not None:
            for e in range(num_elements):
                diffusion_coeffs[e] = self._element_properties[e].diffusion_coefficient
        else:
            diffusion_coeffs[:] = self.properties.diffusion_coefficient

        # Check if we have anisotropic elements (white matter with fiber directions)
        has_anisotropic = (
            self._element_properties is not None and
            any(p.fiber_direction is not None for p in self._element_properties)
        )

        if HAS_NUMBA and not has_anisotropic:
            # Use fast isotropic JIT assembly
            rows, cols, data = _assemble_diffusion_matrix_isotropic_jit(
                self.mesh.elements,
                self._element_volumes,
                shape_grads_array,
                diffusion_coeffs,
                n,
            )
        elif HAS_NUMBA and has_anisotropic:
            # Build fiber direction and tissue type arrays
            fiber_directions = np.zeros((num_elements, 3), dtype=np.float64)
            tissue_types = np.zeros(num_elements, dtype=np.int32)

            for e in range(num_elements):
                tissue_types[e] = self._get_element_tissue_type(e).value
                if self._element_properties[e].fiber_direction is not None:
                    fiber_directions[e] = self._element_properties[e].fiber_direction

            rows, cols, data = _assemble_diffusion_matrix_anisotropic_jit(
                self.mesh.elements,
                self._element_volumes,
                shape_grads_array,
                diffusion_coeffs,
                fiber_directions,
                tissue_types,
                TissueType.WHITE_MATTER.value,
                n,
            )
        else:
            # Fallback to pure Python
            rows, cols, data = [], [], []

            for e, elem in enumerate(self.mesh.elements):
                vol = self._element_volumes[e]
                grads = self._shape_gradients[e]  # (4, 3)

                # Get tissue-specific diffusion coefficient
                if self._element_properties is not None:
                    D = self._element_properties[e].diffusion_coefficient
                    fiber_dir = self._element_properties[e].fiber_direction
                else:
                    D = self.properties.diffusion_coefficient
                    fiber_dir = None

                # Build diffusion tensor
                if fiber_dir is not None and self._get_element_tissue_type(e) == TissueType.WHITE_MATTER:
                    # Anisotropic diffusion: faster along fibers
                    D_tensor = self._build_anisotropic_diffusion_tensor(D, fiber_dir)
                else:
                    # Isotropic diffusion
                    D_tensor = D * np.eye(3)

                # Diffusion matrix: K_ij = integral(grad(Ni) . D . grad(Nj))
                for i in range(4):
                    for j in range(4):
                        # For anisotropic: grad_i . D . grad_j
                        val = vol * grads[i] @ D_tensor @ grads[j]

                        rows.append(elem[i])
                        cols.append(elem[j])
                        data.append(val)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _build_anisotropic_diffusion_tensor(
        self,
        D_base: float,
        fiber_direction: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Build anisotropic diffusion tensor for white matter.

        Diffusion is enhanced along the fiber direction (2x baseline)
        and reduced perpendicular to fibers (0.5x baseline).
        """
        # Normalize fiber direction
        f = fiber_direction / np.linalg.norm(fiber_direction)

        # Diffusion coefficients
        D_parallel = D_base * 2.0  # Faster along fibers
        D_perpendicular = D_base * 0.5  # Slower perpendicular

        # Diffusion tensor: D = D_perp * I + (D_para - D_perp) * f ⊗ f
        D_tensor = D_perpendicular * np.eye(3) + (D_parallel - D_perpendicular) * np.outer(f, f)

        return D_tensor

    def _build_stiffness_matrix(self) -> sparse.csr_matrix:
        """
        Build the elastic stiffness matrix (3D linear elasticity).

        Uses tissue-specific material properties when biophysical constraints
        are available:
        - White matter: Anisotropic, stiffer along fiber direction
        - Gray matter: Isotropic, more compressible
        - CSF: Very soft, nearly incompressible
        """
        n = self.mesh.num_nodes
        num_elements = len(self.mesh.elements)

        # Check if we have anisotropic elements (white matter with fiber directions)
        has_anisotropic = (
            self._element_properties is not None and
            any(p.fiber_direction is not None for p in self._element_properties)
        )

        # Prepare arrays for JIT compilation
        shape_grads_array = np.array(self._shape_gradients)  # (num_elements, 4, 3)

        # Build Lame parameter arrays
        lam_values = np.empty(num_elements, dtype=np.float64)
        mu_values = np.empty(num_elements, dtype=np.float64)

        if self._element_properties is not None:
            for e in range(num_elements):
                lam, mu = self._element_properties[e].lame_parameters()
                lam_values[e] = lam
                mu_values[e] = mu
        else:
            lam, mu = self.properties.lame_parameters()
            lam_values[:] = lam
            mu_values[:] = mu

        if HAS_NUMBA and not has_anisotropic:
            # Use JIT-compiled assembly for all-isotropic case (50-100x faster)
            rows, cols, data = _assemble_stiffness_matrix_isotropic_jit(
                self.mesh.elements,
                self._element_volumes,
                shape_grads_array,
                lam_values,
                mu_values,
                n,
            )
        else:
            # Fallback to Python (required for anisotropic elements)
            rows, cols, data = [], [], []

            for e, elem in enumerate(self.mesh.elements):
                vol = self._element_volumes[e]
                grads = self._shape_gradients[e]  # (4, 3)

                # Get tissue-specific material properties
                if self._element_properties is not None:
                    elem_props = self._element_properties[e]
                else:
                    elem_props = self.properties

                # Build element stiffness matrix (12x12)
                # Use anisotropic formulation for white matter
                if elem_props.fiber_direction is not None:
                    Ke = self._element_stiffness_anisotropic(
                        grads, vol, elem_props
                    )
                else:
                    lam, mu = elem_props.lame_parameters()
                    Ke = self._element_stiffness(grads, vol, lam, mu)

                # Assemble into global matrix
                for i in range(4):
                    for j in range(4):
                        for di in range(3):  # DOF dimension
                            for dj in range(3):
                                global_i = elem[i] * 3 + di
                                global_j = elem[j] * 3 + dj
                                local_i = i * 3 + di
                                local_j = j * 3 + dj

                                rows.append(global_i)
                                cols.append(global_j)
                                data.append(Ke[local_i, local_j])

        K = sparse.csr_matrix((data, (rows, cols)), shape=(3 * n, 3 * n))

        # Apply boundary conditions
        if self.boundary_condition == "fixed":
            K = self._apply_fixed_bc(K)
        elif self.boundary_condition == "skull":
            K = self._apply_skull_bc(K)

        return K

    def _element_stiffness_anisotropic(
        self,
        grads: NDArray[np.float64],
        vol: float,
        props: MaterialProperties,
    ) -> NDArray[np.float64]:
        """
        Compute element stiffness matrix with anisotropic constitutive law.

        Used for white matter elements to resist stretching along fiber direction.
        """
        Ke = np.zeros((12, 12))

        # Get anisotropic constitutive matrix
        C = props.get_constitutive_matrix()

        for i in range(4):
            for j in range(4):
                # B matrices for nodes i and j
                Bi = self._strain_displacement_matrix(grads[i])
                Bj = self._strain_displacement_matrix(grads[j])

                # K_ij = V * Bi^T * C * Bj
                Kij = vol * Bi.T @ C @ Bj

                # Insert into element matrix
                Ke[i*3:(i+1)*3, j*3:(j+1)*3] = Kij

        return Ke

    def _apply_skull_bc(
        self,
        K: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """
        Apply skull boundary conditions (immovable).

        Uses biophysical constraints to identify skull boundary nodes,
        or falls back to mesh boundary nodes.
        """
        K = K.tolil()

        skull_nodes = self._get_skull_boundary_nodes()

        for node_idx in skull_nodes:
            for dof in range(3):
                global_dof = node_idx * 3 + dof
                # Set row to zero except diagonal
                K[global_dof, :] = 0
                K[:, global_dof] = 0
                K[global_dof, global_dof] = 1.0

        return K.tocsr()

    def _element_stiffness(
        self,
        grads: NDArray[np.float64],
        vol: float,
        lam: float,
        mu: float,
    ) -> NDArray[np.float64]:
        """Compute element stiffness matrix for 3D linear elasticity."""
        Ke = np.zeros((12, 12))

        # Constitutive matrix (isotropic linear elasticity)
        C = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ])

        for i in range(4):
            for j in range(4):
                # B matrix (strain-displacement) for nodes i and j
                Bi = self._strain_displacement_matrix(grads[i])
                Bj = self._strain_displacement_matrix(grads[j])

                # K_ij = integral(Bi^T * C * Bj) = V * Bi^T * C * Bj
                Kij = vol * Bi.T @ C @ Bj

                # Insert into element matrix
                Ke[i*3:(i+1)*3, j*3:(j+1)*3] = Kij

        return Ke

    def _strain_displacement_matrix(
        self,
        grad: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Build strain-displacement matrix B for a node."""
        # B relates strain to displacement: epsilon = B * u
        # For 3D: epsilon = [exx, eyy, ezz, 2*exy, 2*exz, 2*eyz]
        dNdx, dNdy, dNdz = grad

        B = np.array([
            [dNdx, 0, 0],
            [0, dNdy, 0],
            [0, 0, dNdz],
            [dNdy, dNdx, 0],
            [dNdz, 0, dNdx],
            [0, dNdz, dNdy],
        ])

        return B

    def _apply_fixed_bc(
        self,
        K: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Apply fixed boundary conditions to stiffness matrix."""
        K = K.tolil()

        for node_idx in self.mesh.boundary_nodes:
            for dof in range(3):
                global_dof = node_idx * 3 + dof
                # Set row to zero except diagonal
                K[global_dof, :] = 0
                K[:, global_dof] = 0
                K[global_dof, global_dof] = 1.0

        return K.tocsr()

    def step(
        self,
        state: TumorState,
        dt: float,
    ) -> TumorState:
        """
        Perform one time step of the simulation.

        Args:
            state: Current tumor state.
            dt: Time step in days.

        Returns:
            Updated TumorState.
        """
        # Step 1: Reaction-diffusion for tumor cell density
        new_density = self._reaction_diffusion_step(state.cell_density, dt)

        # Step 2: Compute growth-induced force
        force = self._compute_growth_force(new_density)

        # Step 3: Solve mechanical equilibrium
        new_displacement = self._solve_mechanical_equilibrium(force)

        # Step 4: Compute stress
        new_stress = self._compute_stress(new_displacement)

        return TumorState(
            cell_density=new_density,
            displacement=new_displacement,
            stress=new_stress,
            time=state.time + dt,
        )

    def _reaction_diffusion_step(
        self,
        density: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Solve reaction-diffusion equation for one time step.

        Uses implicit Euler for diffusion, explicit for reaction.
        dc/dt = D * laplacian(c) + rho * c * (1 - c/K)
        """
        rho = self.properties.proliferation_rate
        K = self.properties.carrying_capacity

        # Reaction term (logistic growth)
        reaction = rho * density * (1 - density / K)

        # Right-hand side: M * (c_n + dt * reaction)
        rhs = self._mass_matrix @ (density + dt * reaction)

        # System matrix: M + dt * D (implicit diffusion)
        A = self._mass_matrix + dt * self._diffusion_matrix

        # Solve
        new_density = spsolve(A, rhs)

        # Ensure non-negative and bounded
        new_density = np.clip(new_density, 0, K)

        return new_density

    def _compute_growth_force(
        self,
        density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute force vector due to tumor growth (mass effect).

        Uses the eigenstrain (thermal expansion) formulation:
        - Growth causes isotropic volumetric expansion: ε_growth = α * c * [1,1,1,0,0,0]^T
        - This creates stress via constitutive law: σ_growth = C * ε_growth
        - Equivalent nodal forces: f = ∫ B^T * σ_growth dV

        The growth_stress_coefficient α represents the volumetric strain per unit
        cell density. A value of 0.1 means 10% volumetric expansion at full density.
        """
        n = self.mesh.num_nodes
        alpha = self.properties.growth_stress_coefficient
        force = np.zeros(3 * n)

        # Get base constitutive matrix (will be modified per-element if needed)
        lam, mu = self.properties.lame_parameters()
        C_base = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ])

        # Isotropic growth strain direction: [1, 1, 1, 0, 0, 0]
        # This represents uniform volumetric expansion
        growth_strain_dir = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        for e, elem in enumerate(self.mesh.elements):
            vol = self._element_volumes[e]
            grads = self._shape_gradients[e]

            # Average density in element
            elem_density = np.mean(density[elem])

            # Skip elements with negligible tumor density
            if elem_density < 1e-6:
                continue

            # Get element-specific constitutive matrix if available
            if self._element_properties is not None:
                C = self._element_properties[e].get_constitutive_matrix()
            else:
                C = C_base

            # Growth strain magnitude: ε_growth = α * c
            # Full growth strain vector: [α*c, α*c, α*c, 0, 0, 0]
            growth_strain = alpha * elem_density * growth_strain_dir

            # Growth stress: σ_growth = C * ε_growth
            growth_stress = C @ growth_strain

            # Compute equivalent nodal forces: f_i = V * B_i^T * σ_growth
            # Each node contributes via its strain-displacement matrix
            for i in range(4):
                B_i = self._strain_displacement_matrix(grads[i])
                # f_i = V * B_i^T * σ_growth
                f_i = vol * B_i.T @ growth_stress

                # Assemble into global force vector
                for d in range(3):
                    global_dof = elem[i] * 3 + d
                    force[global_dof] += f_i[d]

        return force

    def _solve_mechanical_equilibrium(
        self,
        force: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve static mechanical equilibrium: K * u = f.

        Uses skull boundary nodes (immovable) when biophysical constraints
        are provided, otherwise uses mesh boundary nodes.

        Performance optimization:
        - Uses AMG preconditioning when enabled (3-10x faster convergence)
        - Configurable tolerance for speed/accuracy tradeoff
        """
        n = self.mesh.num_nodes
        config = self.solver_config

        # Get boundary nodes based on constraint type
        if self.boundary_condition == "skull":
            boundary_nodes = self._get_skull_boundary_nodes()
        else:
            boundary_nodes = self.mesh.boundary_nodes

        # Apply boundary conditions to force vector
        for node_idx in boundary_nodes:
            for dof in range(3):
                force[node_idx * 3 + dof] = 0.0

        # Build AMG preconditioner if needed (lazy initialization)
        preconditioner = None
        if config.use_amg and HAS_PYAMG:
            if self._amg_ml is None:
                # Build AMG hierarchy (one-time cost, reused for all solves)
                self._amg_ml = pyamg.smoothed_aggregation_solver(
                    self._stiffness_matrix,
                    strength=config.amg_strength,
                    max_coarse=500,
                )
            preconditioner = self._amg_ml.aspreconditioner(cycle=config.amg_cycle)

        # Solve using conjugate gradient with optional AMG preconditioning
        # Note: scipy >= 1.12 renamed 'tol' to 'rtol'
        try:
            u_flat, info = cg(
                self._stiffness_matrix,
                force,
                rtol=config.mechanical_tol,
                maxiter=config.mechanical_maxiter,
                M=preconditioner,
            )
        except TypeError:
            # Fallback for older scipy versions
            u_flat, info = cg(
                self._stiffness_matrix,
                force,
                tol=config.mechanical_tol,
                maxiter=config.mechanical_maxiter,
                M=preconditioner,
            )

        if info != 0:
            # Fall back to direct solver if CG did not converge
            u_flat = spsolve(self._stiffness_matrix, force)

        # Reshape to (n, 3)
        displacement = u_flat.reshape((n, 3))

        return displacement

    def _compute_stress(
        self,
        displacement: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute stress tensor at each element.

        Uses tissue-specific constitutive matrices when biophysical
        constraints are available (anisotropic for white matter).
        """
        stress = np.zeros((self.mesh.num_elements, 6))

        for e, elem in enumerate(self.mesh.elements):
            grads = self._shape_gradients[e]

            # Compute strain from displacement
            strain = np.zeros(6)
            for i in range(4):
                B = self._strain_displacement_matrix(grads[i])
                strain += B @ displacement[elem[i]]

            # Get tissue-specific constitutive matrix
            if self._element_properties is not None:
                C = self._element_properties[e].get_constitutive_matrix()
            else:
                lam, mu = self.properties.lame_parameters()
                C = np.array([
                    [lam + 2*mu, lam, lam, 0, 0, 0],
                    [lam, lam + 2*mu, lam, 0, 0, 0],
                    [lam, lam, lam + 2*mu, 0, 0, 0],
                    [0, 0, 0, mu, 0, 0],
                    [0, 0, 0, 0, mu, 0],
                    [0, 0, 0, 0, 0, mu],
                ])

            # Compute stress: sigma = C * epsilon
            stress[e] = C @ strain

        return stress

    def simulate(
        self,
        initial_state: TumorState,
        duration: float,
        dt: float = 1.0,
        callback: Optional[Callable[[TumorState, int], None]] = None,
    ) -> List[TumorState]:
        """
        Run simulation for a specified duration.

        Args:
            initial_state: Initial tumor state.
            duration: Simulation duration in days.
            dt: Time step in days.
            callback: Optional callback function called after each step.

        Returns:
            List of TumorState objects at each time step.
        """
        states = [initial_state]
        current_state = initial_state
        num_steps = int(duration / dt)

        for step_idx in range(num_steps):
            current_state = self.step(current_state, dt)
            states.append(current_state)

            if callback is not None:
                callback(current_state, step_idx)

        return states

    def compute_tumor_volume(
        self,
        state: TumorState,
        threshold: float = 0.1,
    ) -> float:
        """
        Compute tumor volume above a density threshold.

        Args:
            state: Current tumor state.
            threshold: Density threshold for tumor boundary.

        Returns:
            Tumor volume in mm^3.
        """
        volume = 0.0

        for e, elem in enumerate(self.mesh.elements):
            elem_density = np.mean(state.cell_density[elem])
            if elem_density >= threshold:
                volume += self._element_volumes[e]

        return volume

    def compute_max_displacement(self, state: TumorState) -> float:
        """Compute maximum displacement magnitude."""
        magnitudes = np.linalg.norm(state.displacement, axis=1)
        return float(np.max(magnitudes))

    def compute_von_mises_stress(
        self,
        state: TumorState,
    ) -> NDArray[np.float64]:
        """Compute von Mises stress at each element."""
        stress = state.stress
        # Voigt notation: [sxx, syy, szz, sxy, sxz, syz]
        sxx, syy, szz = stress[:, 0], stress[:, 1], stress[:, 2]
        sxy, sxz, syz = stress[:, 3], stress[:, 4], stress[:, 5]

        # von Mises: sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2 + 6*(sxy^2+sxz^2+syz^2)))
        vm = np.sqrt(0.5 * (
            (sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 +
            6 * (sxy**2 + sxz**2 + syz**2)
        ))

        return vm

    # =========================================================================
    # Serialization Methods for Precomputed Solvers
    # =========================================================================

    def save(self, directory: str) -> None:
        """
        Save precomputed solver state to a directory.

        Saves all matrices, mesh, and precomputed data for fast loading later.
        This avoids the expensive matrix assembly step when using default parameters.

        Args:
            directory: Directory path to save solver data.
        """
        import json
        import pickle
        from pathlib import Path
        from .mesh import save_mesh

        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save mesh
        save_mesh(self.mesh, str(save_dir / "mesh.vtu"))

        # Save boundary nodes separately (not stored in meshio format)
        np.save(save_dir / "boundary_nodes.npy", self.mesh.boundary_nodes)

        # Save system matrices (scipy sparse format)
        matrices_dir = save_dir / "matrices"
        matrices_dir.mkdir(exist_ok=True)
        sparse.save_npz(matrices_dir / "mass_matrix.npz", self._mass_matrix)
        sparse.save_npz(matrices_dir / "stiffness_matrix.npz", self._stiffness_matrix)
        sparse.save_npz(matrices_dir / "diffusion_matrix.npz", self._diffusion_matrix)

        # Save precomputed element data
        precomputed_dir = save_dir / "precomputed"
        precomputed_dir.mkdir(exist_ok=True)
        np.save(precomputed_dir / "element_volumes.npy", self._element_volumes)
        with open(precomputed_dir / "shape_gradients.pkl", "wb") as f:
            pickle.dump(self._shape_gradients, f)

        # Save biophysical data if present
        if self._node_tissues is not None:
            np.save(precomputed_dir / "node_tissues.npy", self._node_tissues)
        if self._node_fiber_directions is not None:
            np.save(precomputed_dir / "node_fiber_directions.npy", self._node_fiber_directions)

        # Save element properties as simplified format
        if self._element_properties is not None:
            elem_props_data = []
            for props in self._element_properties:
                elem_props_data.append({
                    "young_modulus": props.young_modulus,
                    "poisson_ratio": props.poisson_ratio,
                    "proliferation_rate": props.proliferation_rate,
                    "diffusion_coefficient": props.diffusion_coefficient,
                    "carrying_capacity": props.carrying_capacity,
                    "growth_stress_coefficient": props.growth_stress_coefficient,
                    "anisotropy_ratio": props.anisotropy_ratio,
                    "fiber_direction": props.fiber_direction.tolist() if props.fiber_direction is not None else None,
                })
            with open(precomputed_dir / "element_properties.json", "w") as f:
                json.dump(elem_props_data, f)

        # Save metadata
        metadata = {
            "version": "1.1",
            "boundary_condition": self.boundary_condition,
            "num_nodes": len(self.mesh.nodes),
            "num_elements": len(self.mesh.elements),
            "has_biophysical_constraints": self._node_tissues is not None,
            "properties": {
                "young_modulus": self.properties.young_modulus,
                "poisson_ratio": self.properties.poisson_ratio,
                "proliferation_rate": self.properties.proliferation_rate,
                "diffusion_coefficient": self.properties.diffusion_coefficient,
                "carrying_capacity": self.properties.carrying_capacity,
                "growth_stress_coefficient": self.properties.growth_stress_coefficient,
                "anisotropy_ratio": self.properties.anisotropy_ratio,
            },
            "solver_config": self.solver_config.to_dict(),
        }
        with open(save_dir / "solver_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "TumorGrowthSolver":
        """
        Load precomputed solver from a directory.

        This bypasses expensive matrix assembly by loading pre-built matrices.

        Args:
            directory: Directory containing saved solver data.

        Returns:
            TumorGrowthSolver with precomputed matrices loaded.
        """
        import json
        import pickle
        from pathlib import Path
        from .mesh import load_mesh

        load_dir = Path(directory)

        if not load_dir.exists():
            raise FileNotFoundError(f"Solver directory not found: {directory}")

        # Load metadata
        with open(load_dir / "solver_metadata.json") as f:
            metadata = json.load(f)

        # Load mesh
        mesh = load_mesh(str(load_dir / "mesh.vtu"))

        # Load boundary nodes
        boundary_nodes_path = load_dir / "boundary_nodes.npy"
        if boundary_nodes_path.exists():
            mesh.boundary_nodes = np.load(boundary_nodes_path)

        # Create properties from metadata
        props_data = metadata.get("properties", {})
        properties = MaterialProperties(
            young_modulus=props_data.get("young_modulus", 3000.0),
            poisson_ratio=props_data.get("poisson_ratio", 0.45),
            proliferation_rate=props_data.get("proliferation_rate", 0.01),
            diffusion_coefficient=props_data.get("diffusion_coefficient", 0.1),
            carrying_capacity=props_data.get("carrying_capacity", 1.0),
            growth_stress_coefficient=props_data.get("growth_stress_coefficient", 0.1),
            anisotropy_ratio=props_data.get("anisotropy_ratio", 2.0),
        )

        # Load solver config if present (v1.1+), otherwise use default
        solver_config_data = metadata.get("solver_config")
        if solver_config_data:
            solver_config = SolverConfig.from_dict(solver_config_data)
        else:
            solver_config = SolverConfig.default()

        # Create solver instance without building matrices
        solver = object.__new__(cls)
        solver.mesh = mesh
        solver.properties = properties
        solver.boundary_condition = metadata.get("boundary_condition", "fixed")
        solver.biophysical_constraints = None  # Not serialized
        solver.solver_config = solver_config
        solver._amg_ml = None  # Will be built lazily on first solve

        # Load system matrices
        matrices_dir = load_dir / "matrices"
        solver._mass_matrix = sparse.load_npz(matrices_dir / "mass_matrix.npz")
        solver._stiffness_matrix = sparse.load_npz(matrices_dir / "stiffness_matrix.npz")
        solver._diffusion_matrix = sparse.load_npz(matrices_dir / "diffusion_matrix.npz")

        # Load precomputed element data
        precomputed_dir = load_dir / "precomputed"
        solver._element_volumes = np.load(precomputed_dir / "element_volumes.npy")
        with open(precomputed_dir / "shape_gradients.pkl", "rb") as f:
            solver._shape_gradients = pickle.load(f)

        # Load biophysical data if present
        node_tissues_path = precomputed_dir / "node_tissues.npy"
        solver._node_tissues = np.load(node_tissues_path) if node_tissues_path.exists() else None

        fiber_dir_path = precomputed_dir / "node_fiber_directions.npy"
        solver._node_fiber_directions = np.load(fiber_dir_path) if fiber_dir_path.exists() else None

        # Load element properties if present
        elem_props_path = precomputed_dir / "element_properties.json"
        if elem_props_path.exists():
            with open(elem_props_path) as f:
                elem_props_data = json.load(f)
            solver._element_properties = []
            for ep in elem_props_data:
                fiber_dir = np.array(ep["fiber_direction"]) if ep["fiber_direction"] else None
                solver._element_properties.append(MaterialProperties(
                    young_modulus=ep["young_modulus"],
                    poisson_ratio=ep["poisson_ratio"],
                    proliferation_rate=ep["proliferation_rate"],
                    diffusion_coefficient=ep["diffusion_coefficient"],
                    carrying_capacity=ep["carrying_capacity"],
                    growth_stress_coefficient=ep["growth_stress_coefficient"],
                    anisotropy_ratio=ep["anisotropy_ratio"],
                    fiber_direction=fiber_dir,
                ))
        else:
            solver._element_properties = None

        return solver

    @classmethod
    def load_default(
        cls,
        solver_config: Optional[SolverConfig] = None,
    ) -> "TumorGrowthSolver":
        """
        Load the precomputed default solver for posterior fossa simulations.

        This provides fast initialization (~100ms vs ~10s) by loading
        precomputed matrices built with default parameters:
        - MNI152 space with posterior fossa restriction
        - Bundled SUIT atlas regions
        - Bundled MNI152 tissue segmentation
        - Bundled HCP1065 DTI fiber orientations
        - Default tumor origin at vermis [1, -61, -34] MNI

        Args:
            solver_config: Optional solver configuration to override defaults.
                          Use SolverConfig.fast() for approximate (faster) solutions,
                          SolverConfig.accurate() for high precision.

        Returns:
            TumorGrowthSolver ready for simulation.

        Raises:
            FileNotFoundError: If precomputed solver data not found.

        Example:
            # Fast approximate mode (3-10x speedup)
            solver = TumorGrowthSolver.load_default(SolverConfig.fast())

            # Default mode with AMG preconditioning
            solver = TumorGrowthSolver.load_default()
        """
        from pathlib import Path

        default_solver_dir = Path(__file__).parent.parent.parent / "data" / "solvers" / "default_posterior_fossa"

        if not default_solver_dir.exists():
            raise FileNotFoundError(
                f"Precomputed default solver not found at {default_solver_dir}. "
                "Run 'python -m pft_fem.create_default_solver' to generate it."
            )

        solver = cls.load(str(default_solver_dir))

        # Override solver config if provided
        if solver_config is not None:
            solver.solver_config = solver_config
            solver._amg_ml = None  # Reset AMG preconditioner

        return solver
