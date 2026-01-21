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

from .mesh import TetMesh

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

    # Mechanical coupling
    growth_stress_coefficient: float = 0.1  # Stress per unit tumor density

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
            growth_stress_coefficient=0.1,
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
            growth_stress_coefficient=0.1,
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
    ):
        """
        Initialize the tumor growth solver.

        Args:
            mesh: Tetrahedral mesh for FEM.
            properties: Material properties (uses defaults if None).
            boundary_condition: Boundary condition type ("fixed", "skull", or "free").
            biophysical_constraints: Optional biophysical constraints for tissue-specific
                                    material properties, fiber orientation, and boundaries.
        """
        self.mesh = mesh
        self.properties = properties or MaterialProperties()
        self.boundary_condition = boundary_condition
        self.biophysical_constraints = biophysical_constraints

        # Tissue and fiber data from biophysical constraints
        self._node_tissues: Optional[NDArray[np.int32]] = None
        self._node_fiber_directions: Optional[NDArray[np.float64]] = None
        self._element_properties: Optional[List[MaterialProperties]] = None

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

        # 3n x 3n matrix (3 DOFs per node: ux, uy, uz)
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
        """Compute force vector due to tumor growth (mass effect)."""
        n = self.mesh.num_nodes
        alpha = self.properties.growth_stress_coefficient
        force = np.zeros(3 * n)

        for e, elem in enumerate(self.mesh.elements):
            vol = self._element_volumes[e]
            grads = self._shape_gradients[e]

            # Average density in element
            elem_density = np.mean(density[elem])

            # Growth force proportional to density gradient
            for i in range(4):
                for d in range(3):
                    global_dof = elem[i] * 3 + d
                    # Force due to isotropic expansion
                    force[global_dof] += alpha * elem_density * vol * grads[i, d]

        return force

    def _solve_mechanical_equilibrium(
        self,
        force: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Solve static mechanical equilibrium: K * u = f.

        Uses skull boundary nodes (immovable) when biophysical constraints
        are provided, otherwise uses mesh boundary nodes.
        """
        n = self.mesh.num_nodes

        # Get boundary nodes based on constraint type
        if self.boundary_condition == "skull":
            boundary_nodes = self._get_skull_boundary_nodes()
        else:
            boundary_nodes = self.mesh.boundary_nodes

        # Apply boundary conditions to force vector
        for node_idx in boundary_nodes:
            for dof in range(3):
                force[node_idx * 3 + dof] = 0.0

        # Solve using conjugate gradient
        # Note: scipy >= 1.12 renamed 'tol' to 'rtol'
        try:
            u_flat, info = cg(self._stiffness_matrix, force, rtol=1e-6, maxiter=1000)
        except TypeError:
            # Fallback for older scipy versions
            u_flat, info = cg(self._stiffness_matrix, force, tol=1e-6, maxiter=1000)

        if info != 0:
            # Fall back to direct solver
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
                callback(current_state, step_idx + 1)

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
