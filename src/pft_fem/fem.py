"""
Finite Element Method solver for tumor growth simulation.

Implements a coupled reaction-diffusion and mechanical deformation model
for simulating tumor growth in brain tissue.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Callable, Dict, List, Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg

from .mesh import TetMesh


class TissueType(Enum):
    """Brain tissue types with different mechanical properties."""

    GRAY_MATTER = "gray_matter"
    WHITE_MATTER = "white_matter"
    CSF = "csf"
    TUMOR = "tumor"
    EDEMA = "edema"


@dataclass
class MaterialProperties:
    """
    Material properties for brain tissue FEM simulation.

    Properties are based on literature values for brain tissue mechanics.
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

    # Tissue-specific multipliers
    tissue_stiffness_multipliers: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 1.0,
        TissueType.WHITE_MATTER: 1.2,  # Slightly stiffer
        TissueType.CSF: 0.01,  # Very soft (fluid)
        TissueType.TUMOR: 2.0,  # Tumors are often stiffer
        TissueType.EDEMA: 0.5,  # Softened tissue
    })

    tissue_diffusion_multipliers: Dict[TissueType, float] = field(default_factory=lambda: {
        TissueType.GRAY_MATTER: 1.0,
        TissueType.WHITE_MATTER: 2.0,  # Faster along fiber tracts
        TissueType.CSF: 0.1,  # Barrier to invasion
        TissueType.TUMOR: 0.5,
        TissueType.EDEMA: 1.5,
    })

    @classmethod
    def for_tissue(cls, tissue_type: TissueType) -> "MaterialProperties":
        """Create material properties for a specific tissue type."""
        base = cls()
        stiffness_mult = base.tissue_stiffness_multipliers.get(tissue_type, 1.0)
        diffusion_mult = base.tissue_diffusion_multipliers.get(tissue_type, 1.0)

        return cls(
            young_modulus=base.young_modulus * stiffness_mult,
            poisson_ratio=base.poisson_ratio,
            proliferation_rate=base.proliferation_rate,
            diffusion_coefficient=base.diffusion_coefficient * diffusion_mult,
            carrying_capacity=base.carrying_capacity,
            growth_stress_coefficient=base.growth_stress_coefficient,
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
    """

    def __init__(
        self,
        mesh: TetMesh,
        properties: Optional[MaterialProperties] = None,
        boundary_condition: str = "fixed",
    ):
        """
        Initialize the tumor growth solver.

        Args:
            mesh: Tetrahedral mesh for FEM.
            properties: Material properties (uses defaults if None).
            boundary_condition: Boundary condition type ("fixed" or "free").
        """
        self.mesh = mesh
        self.properties = properties or MaterialProperties()
        self.boundary_condition = boundary_condition

        # Precompute element matrices
        self._element_volumes = mesh.compute_element_volumes()
        self._shape_gradients = self._compute_shape_gradients()

        # Build system matrices
        self._mass_matrix = self._build_mass_matrix()
        self._stiffness_matrix = self._build_stiffness_matrix()
        self._diffusion_matrix = self._build_diffusion_matrix()

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
        """Build the diffusion matrix."""
        n = self.mesh.num_nodes
        D = self.properties.diffusion_coefficient
        rows, cols, data = [], [], []

        for e, elem in enumerate(self.mesh.elements):
            vol = self._element_volumes[e]
            grads = self._shape_gradients[e]  # (4, 3)

            # Diffusion matrix: K_ij = D * integral(grad(Ni) . grad(Nj))
            # For linear elements: K_ij = D * V * grad(Ni) . grad(Nj)
            for i in range(4):
                for j in range(4):
                    val = D * vol * np.dot(grads[i], grads[j])

                    rows.append(elem[i])
                    cols.append(elem[j])
                    data.append(val)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _build_stiffness_matrix(self) -> sparse.csr_matrix:
        """Build the elastic stiffness matrix (3D linear elasticity)."""
        n = self.mesh.num_nodes
        lam, mu = self.properties.lame_parameters()

        # 3n x 3n matrix (3 DOFs per node: ux, uy, uz)
        rows, cols, data = [], [], []

        for e, elem in enumerate(self.mesh.elements):
            vol = self._element_volumes[e]
            grads = self._shape_gradients[e]  # (4, 3)

            # Build element stiffness matrix (12x12)
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

        return K

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
        """Solve static mechanical equilibrium: K * u = f."""
        n = self.mesh.num_nodes

        # Apply boundary conditions to force vector
        for node_idx in self.mesh.boundary_nodes:
            for dof in range(3):
                force[node_idx * 3 + dof] = 0.0

        # Solve
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
        """Compute stress tensor at each element."""
        lam, mu = self.properties.lame_parameters()
        stress = np.zeros((self.mesh.num_elements, 6))

        for e, elem in enumerate(self.mesh.elements):
            grads = self._shape_gradients[e]

            # Compute strain from displacement
            strain = np.zeros(6)
            for i in range(4):
                B = self._strain_displacement_matrix(grads[i])
                strain += B @ displacement[elem[i]]

            # Compute stress: sigma = C * epsilon
            stress[e, 0] = (lam + 2*mu) * strain[0] + lam * (strain[1] + strain[2])
            stress[e, 1] = (lam + 2*mu) * strain[1] + lam * (strain[0] + strain[2])
            stress[e, 2] = (lam + 2*mu) * strain[2] + lam * (strain[0] + strain[1])
            stress[e, 3] = mu * strain[3]
            stress[e, 4] = mu * strain[4]
            stress[e, 5] = mu * strain[5]

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
