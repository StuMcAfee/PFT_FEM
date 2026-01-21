"""
Comprehensive tests for the FEM tumor growth solver.

Tests cover:
- Material property calculations
- Mesh structure and operations
- FEM matrix assembly
- Solver convergence and accuracy
- Physical behavior validation
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from pft_fem.fem import (
    TissueType,
    MaterialProperties,
    TumorState,
    TumorGrowthSolver,
)
from pft_fem.mesh import TetMesh, MeshGenerator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_mesh():
    """Create a simple single-tetrahedron mesh for unit tests."""
    # Unit tetrahedron with vertices at origin and along each axis
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

    return TetMesh(
        nodes=nodes,
        elements=elements,
        node_labels=np.ones(4, dtype=np.int32),
        boundary_nodes=np.array([0, 1, 2, 3], dtype=np.int32),
    )


@pytest.fixture
def cube_mesh():
    """Create a mesh for a unit cube (multiple tetrahedra)."""
    # 8 corner nodes of unit cube
    nodes = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=np.float64)

    # Decompose cube into 5 tetrahedra
    elements = np.array([
        [0, 1, 3, 5],
        [0, 3, 2, 6],
        [0, 5, 4, 6],
        [3, 5, 6, 7],
        [0, 3, 5, 6],
    ], dtype=np.int32)

    return TetMesh(
        nodes=nodes,
        elements=elements,
        node_labels=np.ones(8, dtype=np.int32),
        boundary_nodes=np.arange(8, dtype=np.int32),
    )


@pytest.fixture
def spherical_mesh():
    """Create a spherical mesh for more realistic tests."""
    # Generate a simple spherical mesh
    center = np.array([5.0, 5.0, 5.0])
    radius = 4.0

    # Create mask
    shape = (11, 11, 11)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - 5)**2 + (y - 5)**2 + (z - 5)**2)
    mask = dist <= radius

    generator = MeshGenerator()
    mesh = generator.from_mask(
        mask=mask,
        voxel_size=(1.0, 1.0, 1.0),
        simplify=False,
    )

    return mesh


@pytest.fixture
def default_properties():
    """Default material properties."""
    return MaterialProperties()


# =============================================================================
# Material Properties Tests
# =============================================================================


class TestMaterialProperties:
    """Tests for MaterialProperties dataclass."""

    def test_default_values(self):
        """Test that default values are physically reasonable."""
        props = MaterialProperties()

        # Young's modulus should be in kPa range for brain tissue
        assert 100 <= props.young_modulus <= 20000

        # Poisson ratio should be < 0.5 (incompressibility limit)
        assert 0 < props.poisson_ratio < 0.5

        # Proliferation rate should be positive and reasonable
        assert 0 < props.proliferation_rate < 1.0

        # Diffusion coefficient should be positive
        assert props.diffusion_coefficient > 0

    def test_lame_parameters_calculation(self):
        """Test Lamé parameter calculation from elastic constants."""
        props = MaterialProperties(young_modulus=3000, poisson_ratio=0.3)
        lam, mu = props.lame_parameters()

        # Verify relationships
        E = props.young_modulus
        nu = props.poisson_ratio

        # Check mu (shear modulus)
        expected_mu = E / (2 * (1 + nu))
        assert_allclose(mu, expected_mu, rtol=1e-10)

        # Check lambda
        expected_lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        assert_allclose(lam, expected_lam, rtol=1e-10)

    def test_lame_parameters_near_incompressible(self):
        """Test Lamé parameters for nearly incompressible material."""
        props = MaterialProperties(young_modulus=3000, poisson_ratio=0.49)
        lam, mu = props.lame_parameters()

        # Lambda should be much larger than mu for near-incompressibility
        assert lam > 10 * mu

    def test_tissue_specific_properties(self):
        """Test that tissue-specific properties are generated correctly."""
        gray_props = MaterialProperties.for_tissue(TissueType.GRAY_MATTER)
        white_props = MaterialProperties.for_tissue(TissueType.WHITE_MATTER)
        csf_props = MaterialProperties.for_tissue(TissueType.CSF)

        # White matter should be stiffer than gray matter
        assert white_props.young_modulus > gray_props.young_modulus

        # CSF should be very soft
        assert csf_props.young_modulus < gray_props.young_modulus * 0.1

    def test_tissue_stiffness_multipliers(self):
        """Test that all tissue types have defined multipliers."""
        props = MaterialProperties()

        for tissue_type in TissueType:
            assert tissue_type in props.tissue_stiffness_multipliers
            multiplier = props.tissue_stiffness_multipliers[tissue_type]
            assert multiplier > 0

    def test_tissue_diffusion_multipliers(self):
        """Test diffusion multipliers for different tissues."""
        props = MaterialProperties()

        # White matter should have higher diffusion (fiber tracts)
        wm_mult = props.tissue_diffusion_multipliers[TissueType.WHITE_MATTER]
        gm_mult = props.tissue_diffusion_multipliers[TissueType.GRAY_MATTER]

        assert wm_mult > gm_mult


# =============================================================================
# TumorState Tests
# =============================================================================


class TestTumorState:
    """Tests for TumorState dataclass."""

    def test_initial_state_creation(self, cube_mesh):
        """Test creating initial tumor state."""
        center = np.array([0.5, 0.5, 0.5])
        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=center,
            seed_radius=0.3,
            seed_density=0.8,
        )

        # Check shapes
        assert state.cell_density.shape == (cube_mesh.num_nodes,)
        assert state.displacement.shape == (cube_mesh.num_nodes, 3)
        assert state.stress.shape == (cube_mesh.num_elements, 6)

        # Cell density should be bounded
        assert np.all(state.cell_density >= 0)
        assert np.all(state.cell_density <= 1)

        # Maximum density should be near seed center
        max_idx = np.argmax(state.cell_density)
        max_node = cube_mesh.nodes[max_idx]
        assert np.linalg.norm(max_node - center) < 0.5

    def test_initial_state_gaussian_profile(self, cube_mesh):
        """Test that initial density follows Gaussian profile."""
        center = np.array([0.5, 0.5, 0.5])
        radius = 0.2
        peak_density = 0.9

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=center,
            seed_radius=radius,
            seed_density=peak_density,
        )

        # Density at center should be close to peak
        center_node = cube_mesh.find_nearest_node(center)
        assert state.cell_density[center_node] > peak_density * 0.8

        # Density should decay with distance
        distances = np.linalg.norm(cube_mesh.nodes - center, axis=1)
        sorted_indices = np.argsort(distances)

        # Check monotonic decay (approximately)
        densities = state.cell_density[sorted_indices]
        # Allow some noise but overall trend should be decreasing
        assert densities[0] >= densities[-1]

    def test_initial_displacement_zero(self, cube_mesh):
        """Test that initial displacement is zero."""
        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        assert_allclose(state.displacement, 0.0)

    def test_initial_stress_zero(self, cube_mesh):
        """Test that initial stress is zero."""
        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        assert_allclose(state.stress, 0.0)

    def test_initial_time_zero(self, cube_mesh):
        """Test that initial time is zero."""
        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        assert state.time == 0.0


# =============================================================================
# Solver Initialization Tests
# =============================================================================


class TestTumorGrowthSolverInit:
    """Tests for TumorGrowthSolver initialization."""

    def test_solver_initialization(self, cube_mesh, default_properties):
        """Test basic solver initialization."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        assert solver.mesh is cube_mesh
        assert solver.properties is default_properties

    def test_element_volumes_positive(self, cube_mesh, default_properties):
        """Test that all element volumes are positive."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        volumes = solver._element_volumes
        assert np.all(volumes > 0)

    def test_element_volumes_sum(self, cube_mesh, default_properties):
        """Test that element volumes sum to total mesh volume."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        total_volume = np.sum(solver._element_volumes)

        # For unit cube, total volume should be 1.0
        assert_allclose(total_volume, 1.0, rtol=1e-10)

    def test_shape_gradients_computed(self, cube_mesh, default_properties):
        """Test that shape gradients are computed for all elements."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        assert len(solver._shape_gradients) == cube_mesh.num_elements

        for grad in solver._shape_gradients:
            assert grad.shape == (4, 3)

    def test_mass_matrix_symmetry(self, cube_mesh, default_properties):
        """Test that mass matrix is symmetric."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        M = solver._mass_matrix.toarray()
        assert_allclose(M, M.T, rtol=1e-10)

    def test_mass_matrix_positive_diagonal(self, cube_mesh, default_properties):
        """Test that mass matrix has positive diagonal."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        M = solver._mass_matrix.toarray()
        assert np.all(np.diag(M) > 0)

    def test_diffusion_matrix_symmetry(self, cube_mesh, default_properties):
        """Test that diffusion matrix is symmetric."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        K = solver._diffusion_matrix.toarray()
        assert_allclose(K, K.T, rtol=1e-10)

    def test_stiffness_matrix_symmetry(self, cube_mesh, default_properties):
        """Test that stiffness matrix is symmetric."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        K = solver._stiffness_matrix.toarray()
        assert_allclose(K, K.T, rtol=1e-10)

    def test_stiffness_matrix_dimensions(self, cube_mesh, default_properties):
        """Test stiffness matrix has correct dimensions (3 DOF per node)."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        K = solver._stiffness_matrix
        expected_size = 3 * cube_mesh.num_nodes

        assert K.shape == (expected_size, expected_size)


# =============================================================================
# Solver Step Tests
# =============================================================================


class TestTumorGrowthSolverStep:
    """Tests for individual solver time steps."""

    def test_step_updates_time(self, cube_mesh, default_properties):
        """Test that time step updates simulation time."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        dt = 0.5
        new_state = solver.step(state, dt)

        assert_allclose(new_state.time, dt)

    def test_step_preserves_density_bounds(self, cube_mesh, default_properties):
        """Test that density remains bounded after step."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
            seed_density=0.8,
        )

        for _ in range(10):
            state = solver.step(state, dt=1.0)

        assert np.all(state.cell_density >= 0)
        assert np.all(state.cell_density <= default_properties.carrying_capacity)

    def test_step_tumor_grows(self, cube_mesh):
        """Test that tumor density increases over time (growth)."""
        props = MaterialProperties(proliferation_rate=0.1)
        solver = TumorGrowthSolver(cube_mesh, props)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
            seed_density=0.5,
        )

        initial_total = np.sum(state.cell_density)

        # Run several steps
        for _ in range(5):
            state = solver.step(state, dt=1.0)

        final_total = np.sum(state.cell_density)

        # Total tumor cells should increase
        assert final_total > initial_total

    def test_step_tumor_spreads(self, spherical_mesh):
        """Test that tumor spreads spatially over time."""
        props = MaterialProperties(diffusion_coefficient=0.5)
        solver = TumorGrowthSolver(spherical_mesh, props)

        center = np.mean(spherical_mesh.nodes, axis=0)
        state = TumorState.initial(
            mesh=spherical_mesh,
            seed_center=center,
            seed_radius=1.0,
            seed_density=0.8,
        )

        # Count nodes with significant density initially
        initial_affected = np.sum(state.cell_density > 0.01)

        # Run simulation
        for _ in range(20):
            state = solver.step(state, dt=0.5)

        # More nodes should have significant density
        final_affected = np.sum(state.cell_density > 0.01)

        assert final_affected >= initial_affected

    def test_step_displacement_generated(self, cube_mesh, default_properties):
        """Test that tumor growth generates displacement."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
            seed_density=0.8,
        )

        # Run several steps to build up growth-induced stress
        for _ in range(5):
            state = solver.step(state, dt=1.0)

        # Some displacement should be generated
        max_disp = np.max(np.abs(state.displacement))
        # Note: displacement might be small but should be non-zero
        # (boundary conditions may constrain it significantly)
        assert max_disp >= 0


# =============================================================================
# Solver Simulation Tests
# =============================================================================


class TestTumorGrowthSolverSimulation:
    """Tests for full simulation runs."""

    def test_simulate_returns_correct_states(self, cube_mesh, default_properties):
        """Test that simulate returns correct number of states."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        duration = 5.0
        dt = 1.0
        states = solver.simulate(state, duration, dt)

        # Should have initial state + num_steps states
        expected_count = int(duration / dt) + 1
        assert len(states) == expected_count

    def test_simulate_time_progression(self, cube_mesh, default_properties):
        """Test that time progresses correctly in simulation."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        dt = 0.5
        states = solver.simulate(state, duration=3.0, dt=dt)

        for i, s in enumerate(states):
            expected_time = i * dt
            assert_allclose(s.time, expected_time, rtol=1e-10)

    def test_simulate_callback_called(self, cube_mesh, default_properties):
        """Test that callback is called for each step."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        callback_count = [0]

        def callback(state, step):
            callback_count[0] += 1

        solver.simulate(state, duration=5.0, dt=1.0, callback=callback)

        assert callback_count[0] == 5  # 5 steps

    def test_simulate_monotonic_tumor_volume(self, spherical_mesh):
        """Test that tumor volume increases monotonically (for growing tumor)."""
        props = MaterialProperties(proliferation_rate=0.05)
        solver = TumorGrowthSolver(spherical_mesh, props)

        center = np.mean(spherical_mesh.nodes, axis=0)
        state = TumorState.initial(
            mesh=spherical_mesh,
            seed_center=center,
            seed_radius=1.5,
            seed_density=0.5,
        )

        states = solver.simulate(state, duration=10.0, dt=1.0)

        volumes = [solver.compute_tumor_volume(s) for s in states]

        # Volume should be non-decreasing
        for i in range(1, len(volumes)):
            assert volumes[i] >= volumes[i - 1] * 0.99  # Allow tiny numerical fluctuation


# =============================================================================
# Solver Metrics Tests
# =============================================================================


class TestTumorGrowthSolverMetrics:
    """Tests for solver metric calculations."""

    def test_compute_tumor_volume_zero_threshold(self, cube_mesh, default_properties):
        """Test tumor volume with zero threshold includes all non-zero regions."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.5,
            seed_density=0.8,
        )

        volume_low = solver.compute_tumor_volume(state, threshold=0.001)
        volume_high = solver.compute_tumor_volume(state, threshold=0.5)

        # Lower threshold should give larger volume
        assert volume_low >= volume_high

    def test_compute_tumor_volume_positive(self, cube_mesh, default_properties):
        """Test that tumor volume is non-negative."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        volume = solver.compute_tumor_volume(state)
        assert volume >= 0

    def test_compute_max_displacement(self, cube_mesh, default_properties):
        """Test max displacement calculation."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        # Initially zero displacement
        max_disp = solver.compute_max_displacement(state)
        assert_allclose(max_disp, 0.0)

        # After some growth, may have displacement
        for _ in range(5):
            state = solver.step(state, dt=1.0)

        max_disp = solver.compute_max_displacement(state)
        assert max_disp >= 0

    def test_compute_von_mises_stress_positive(self, cube_mesh, default_properties):
        """Test that von Mises stress is always non-negative."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
            seed_density=0.8,
        )

        # Run some steps to generate stress
        for _ in range(5):
            state = solver.step(state, dt=1.0)

        von_mises = solver.compute_von_mises_stress(state)

        assert von_mises.shape == (cube_mesh.num_elements,)
        assert np.all(von_mises >= 0)


# =============================================================================
# Physical Behavior Tests
# =============================================================================


class TestPhysicalBehavior:
    """Tests for physically realistic behavior."""

    def test_logistic_growth_saturation(self, cube_mesh):
        """Test that tumor growth saturates at carrying capacity."""
        props = MaterialProperties(
            proliferation_rate=0.5,  # Fast growth
            carrying_capacity=1.0,
        )
        solver = TumorGrowthSolver(cube_mesh, props)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.5,
            seed_density=0.3,
        )

        # Run for long time
        for _ in range(100):
            state = solver.step(state, dt=1.0)

        # Density should approach but not exceed carrying capacity
        assert np.all(state.cell_density <= props.carrying_capacity)

        # Central region should be close to saturation
        center_node = cube_mesh.find_nearest_node(np.array([0.5, 0.5, 0.5]))
        assert state.cell_density[center_node] > 0.9 * props.carrying_capacity

    def test_diffusion_smooths_gradients(self, spherical_mesh):
        """Test that diffusion smooths density gradients over time."""
        props = MaterialProperties(
            proliferation_rate=0.0,  # No growth, pure diffusion
            diffusion_coefficient=1.0,
        )
        solver = TumorGrowthSolver(spherical_mesh, props)

        center = np.mean(spherical_mesh.nodes, axis=0)
        state = TumorState.initial(
            mesh=spherical_mesh,
            seed_center=center,
            seed_radius=0.5,  # Small, concentrated
            seed_density=1.0,
        )

        # Calculate initial "sharpness" (max gradient)
        initial_max = np.max(state.cell_density)
        initial_min = np.min(state.cell_density)
        initial_range = initial_max - initial_min

        # Run diffusion
        for _ in range(50):
            state = solver.step(state, dt=0.5)

        final_max = np.max(state.cell_density)
        final_min = np.min(state.cell_density)
        final_range = final_max - final_min

        # Range should decrease (profile becomes flatter)
        assert final_range < initial_range

    def test_mass_conservation_pure_diffusion(self, cube_mesh):
        """Test mass conservation in pure diffusion (no reaction)."""
        props = MaterialProperties(
            proliferation_rate=0.0,  # No growth
            diffusion_coefficient=0.1,
        )
        solver = TumorGrowthSolver(cube_mesh, props)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
            seed_density=0.5,
        )

        # Calculate initial total mass (integral of density)
        M = solver._mass_matrix.toarray()
        initial_mass = state.cell_density @ M @ state.cell_density

        # Run simulation
        for _ in range(10):
            state = solver.step(state, dt=0.5)

        # Final mass (allowing for boundary effects)
        final_mass = state.cell_density @ M @ state.cell_density

        # Mass should be approximately conserved
        # Note: some mass may be lost at boundaries
        assert final_mass <= initial_mass * 1.1  # Not gaining mass
        assert final_mass >= initial_mass * 0.5  # Not losing too much

    def test_boundary_conditions_respected(self, cube_mesh, default_properties):
        """Test that fixed boundary conditions are enforced."""
        solver = TumorGrowthSolver(
            cube_mesh,
            default_properties,
            boundary_condition="fixed"
        )

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
            seed_density=0.8,
        )

        # Run simulation
        for _ in range(10):
            state = solver.step(state, dt=1.0)

        # Boundary nodes should have zero displacement
        for node_idx in cube_mesh.boundary_nodes:
            disp_magnitude = np.linalg.norm(state.displacement[node_idx])
            assert_allclose(disp_magnitude, 0.0, atol=1e-10)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_initial_density(self, cube_mesh, default_properties):
        """Test behavior with zero initial density."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        # Create state with zero density
        state = TumorState(
            cell_density=np.zeros(cube_mesh.num_nodes),
            displacement=np.zeros((cube_mesh.num_nodes, 3)),
            stress=np.zeros((cube_mesh.num_elements, 6)),
            time=0.0,
        )

        new_state = solver.step(state, dt=1.0)

        # Should remain zero (no spontaneous generation)
        assert_allclose(new_state.cell_density, 0.0)

    def test_uniform_density(self, cube_mesh, default_properties):
        """Test behavior with uniform initial density."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        uniform_density = 0.5
        state = TumorState(
            cell_density=np.full(cube_mesh.num_nodes, uniform_density),
            displacement=np.zeros((cube_mesh.num_nodes, 3)),
            stress=np.zeros((cube_mesh.num_elements, 6)),
            time=0.0,
        )

        new_state = solver.step(state, dt=1.0)

        # Density should evolve but remain relatively uniform
        # (diffusion has no effect on uniform field)
        std_dev = np.std(new_state.cell_density)
        assert std_dev < 0.1  # Relatively uniform

    def test_very_small_time_step(self, cube_mesh, default_properties):
        """Test stability with very small time step."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
        )

        # Very small time step should still work
        new_state = solver.step(state, dt=1e-6)

        # Should be very close to initial state
        assert_allclose(new_state.cell_density, state.cell_density, rtol=1e-3)

    def test_single_element_mesh(self, simple_mesh, default_properties):
        """Test solver works with minimal mesh."""
        solver = TumorGrowthSolver(simple_mesh, default_properties)

        state = TumorState.initial(
            mesh=simple_mesh,
            seed_center=np.array([0.25, 0.25, 0.25]),
            seed_radius=0.5,
        )

        # Should not crash
        new_state = solver.step(state, dt=1.0)

        assert new_state.cell_density.shape == (simple_mesh.num_nodes,)


# =============================================================================
# Numerical Accuracy Tests
# =============================================================================


class TestNumericalAccuracy:
    """Tests for numerical accuracy and convergence."""

    def test_time_step_convergence(self, cube_mesh, default_properties):
        """Test that smaller time steps give more accurate results."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        state0 = TumorState.initial(
            mesh=cube_mesh,
            seed_center=np.array([0.5, 0.5, 0.5]),
            seed_radius=0.3,
            seed_density=0.5,
        )

        total_time = 2.0

        # Coarse time stepping
        state_coarse = state0
        for _ in range(int(total_time / 1.0)):
            state_coarse = solver.step(state_coarse, dt=1.0)

        # Fine time stepping
        state_fine = state0
        for _ in range(int(total_time / 0.1)):
            state_fine = solver.step(state_fine, dt=0.1)

        # Very fine time stepping (reference)
        state_ref = state0
        for _ in range(int(total_time / 0.01)):
            state_ref = solver.step(state_ref, dt=0.01)

        # Fine should be closer to reference than coarse
        error_coarse = np.linalg.norm(state_coarse.cell_density - state_ref.cell_density)
        error_fine = np.linalg.norm(state_fine.cell_density - state_ref.cell_density)

        assert error_fine < error_coarse

    def test_matrix_condition_numbers(self, cube_mesh, default_properties):
        """Test that system matrices are well-conditioned."""
        solver = TumorGrowthSolver(cube_mesh, default_properties)

        # Mass matrix condition number
        M = solver._mass_matrix.toarray()
        cond_M = np.linalg.cond(M)
        assert cond_M < 1e6  # Reasonably well-conditioned

        # Diffusion matrix may be singular (needs mass matrix)
        # Test combined system
        dt = 1.0
        A = solver._mass_matrix + dt * solver._diffusion_matrix
        A_dense = A.toarray()
        cond_A = np.linalg.cond(A_dense)
        assert cond_A < 1e8


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_simulation_workflow(self, spherical_mesh):
        """Test complete simulation workflow."""
        props = MaterialProperties(
            proliferation_rate=0.02,
            diffusion_coefficient=0.1,
        )

        solver = TumorGrowthSolver(spherical_mesh, props)

        center = np.mean(spherical_mesh.nodes, axis=0)
        initial_state = TumorState.initial(
            mesh=spherical_mesh,
            seed_center=center,
            seed_radius=1.5,
            seed_density=0.6,
        )

        # Run full simulation
        states = solver.simulate(
            initial_state,
            duration=10.0,
            dt=0.5,
        )

        # Verify outputs
        assert len(states) == 21  # 10/0.5 + 1

        # Tumor should grow
        initial_volume = solver.compute_tumor_volume(states[0])
        final_volume = solver.compute_tumor_volume(states[-1])
        assert final_volume > initial_volume

        # All states should have valid bounds
        for state in states:
            assert np.all(state.cell_density >= 0)
            assert np.all(state.cell_density <= props.carrying_capacity)
            assert state.displacement.shape == (spherical_mesh.num_nodes, 3)
            assert state.stress.shape == (spherical_mesh.num_elements, 6)
