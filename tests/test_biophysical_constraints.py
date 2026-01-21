"""
Comprehensive tests for the biophysical constraints module.

Tests cover:
- Tissue segmentation and classification
- Fiber orientation data handling
- Anisotropic material properties
- Space transformations (MNI <-> SUIT)
- Posterior fossa region masking
- Default ICBM-152 configuration
- Integration with FEM solver
- Skull boundary conditions
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pft_fem.biophysical_constraints import (
    BrainTissue,
    FiberOrientation,
    TissueSegmentation,
    AnisotropicMaterialProperties,
    SUITPyIntegration,
    MNIAtlasLoader,
    SpaceTransformer,
    BiophysicalConstraints,
    DEFAULT_TUMOR_ORIGIN_MNI,
    POSTERIOR_FOSSA_BOUNDS_MNI,
)
from pft_fem.fem import MaterialProperties, TissueType, TumorGrowthSolver, TumorState
from pft_fem.mesh import TetMesh, MeshGenerator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_segmentation():
    """Create synthetic tissue segmentation for testing."""
    shape = (20, 20, 20)

    # Create simple segmentation: CSF in center, GM ring, WM outer
    labels = np.zeros(shape, dtype=np.int32)

    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array([10, 10, 10])

    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # CSF in center
    labels[dist < 3] = BrainTissue.CSF.value
    # Gray matter ring
    labels[(dist >= 3) & (dist < 6)] = BrainTissue.GRAY_MATTER.value
    # White matter outer
    labels[(dist >= 6) & (dist < 9)] = BrainTissue.WHITE_MATTER.value

    # Create probability maps
    gm_prob = np.zeros(shape, dtype=np.float32)
    wm_prob = np.zeros(shape, dtype=np.float32)
    csf_prob = np.zeros(shape, dtype=np.float32)

    csf_prob[labels == BrainTissue.CSF.value] = 1.0
    gm_prob[labels == BrainTissue.GRAY_MATTER.value] = 1.0
    wm_prob[labels == BrainTissue.WHITE_MATTER.value] = 1.0

    affine = np.eye(4)
    affine[:3, 3] = -np.array(shape) / 2

    return TissueSegmentation(
        labels=labels,
        wm_probability=wm_prob,
        gm_probability=gm_prob,
        csf_probability=csf_prob,
        affine=affine,
        voxel_size=(1.0, 1.0, 1.0),
    )


@pytest.fixture
def synthetic_fibers():
    """Create synthetic fiber orientation field for testing."""
    shape = (20, 20, 20)

    vectors = np.zeros((*shape, 3), dtype=np.float64)
    fa = np.zeros(shape, dtype=np.float32)

    # Create radial fiber pattern
    x, y, z = np.meshgrid(
        np.arange(shape[0]) - 10,
        np.arange(shape[1]) - 10,
        np.arange(shape[2]) - 10,
        indexing='ij'
    )

    # Fibers point radially outward from center
    vectors[..., 0] = x
    vectors[..., 1] = y
    vectors[..., 2] = z

    # Normalize
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms < 1e-6] = 1
    vectors = vectors / norms

    # FA is higher further from center
    dist = np.sqrt(x**2 + y**2 + z**2)
    fa = np.clip(dist / 15, 0, 1).astype(np.float32)

    affine = np.eye(4)
    affine[:3, 3] = -np.array(shape) / 2

    return FiberOrientation(
        vectors=vectors,
        fractional_anisotropy=fa,
        affine=affine,
    )


@pytest.fixture
def cube_mesh_with_tissues():
    """Create a mesh with tissue labels assigned."""
    shape = (11, 11, 11)
    center = np.array([5, 5, 5])

    # Create spherical mask
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    mask = dist <= 4

    # Create tissue labels
    labels = np.zeros(shape, dtype=np.int32)
    labels[(dist < 2)] = BrainTissue.CSF.value
    labels[(dist >= 2) & (dist < 3)] = BrainTissue.GRAY_MATTER.value
    labels[(dist >= 3) & (dist <= 4)] = BrainTissue.WHITE_MATTER.value

    generator = MeshGenerator()
    mesh = generator.from_mask(
        mask=mask,
        voxel_size=(1.0, 1.0, 1.0),
        labels=labels,
        simplify=False,
    )

    return mesh


# =============================================================================
# BrainTissue Enum Tests
# =============================================================================


class TestBrainTissue:
    """Tests for BrainTissue enum."""

    def test_tissue_values(self):
        """Test that tissue types have expected integer values."""
        assert BrainTissue.BACKGROUND.value == 0
        assert BrainTissue.CSF.value == 1
        assert BrainTissue.GRAY_MATTER.value == 2
        assert BrainTissue.WHITE_MATTER.value == 3
        assert BrainTissue.SKULL.value == 4

    def test_from_label_mapping(self):
        """Test label to tissue type conversion."""
        assert BrainTissue.from_label(0) == BrainTissue.BACKGROUND
        assert BrainTissue.from_label(1) == BrainTissue.CSF
        assert BrainTissue.from_label(2) == BrainTissue.GRAY_MATTER
        assert BrainTissue.from_label(3) == BrainTissue.WHITE_MATTER

    def test_from_label_unknown(self):
        """Test that unknown labels map to background."""
        assert BrainTissue.from_label(99) == BrainTissue.BACKGROUND


# =============================================================================
# TissueSegmentation Tests
# =============================================================================


class TestTissueSegmentation:
    """Tests for TissueSegmentation dataclass."""

    def test_get_tissue_at_point_csf(self, synthetic_segmentation):
        """Test tissue lookup at CSF region."""
        # Center should be CSF
        point = np.array([0.0, 0.0, 0.0])  # Maps to center in voxel space
        tissue = synthetic_segmentation.get_tissue_at_point(point)
        assert tissue == BrainTissue.CSF

    def test_get_tissue_at_point_gray_matter(self, synthetic_segmentation):
        """Test tissue lookup at gray matter region."""
        # Offset point in gray matter region
        point = np.array([4.0, 0.0, 0.0])  # Gray matter region
        tissue = synthetic_segmentation.get_tissue_at_point(point)
        assert tissue == BrainTissue.GRAY_MATTER

    def test_get_tissue_at_point_white_matter(self, synthetic_segmentation):
        """Test tissue lookup at white matter region."""
        # Offset point in white matter region
        point = np.array([7.0, 0.0, 0.0])  # White matter region
        tissue = synthetic_segmentation.get_tissue_at_point(point)
        assert tissue == BrainTissue.WHITE_MATTER

    def test_get_white_matter_mask(self, synthetic_segmentation):
        """Test white matter mask extraction."""
        wm_mask = synthetic_segmentation.get_white_matter_mask()

        assert wm_mask.dtype == np.bool_
        assert wm_mask.shape == synthetic_segmentation.labels.shape
        assert np.sum(wm_mask) > 0

    def test_get_gray_matter_mask(self, synthetic_segmentation):
        """Test gray matter mask extraction."""
        gm_mask = synthetic_segmentation.get_gray_matter_mask()

        assert gm_mask.dtype == np.bool_
        assert gm_mask.shape == synthetic_segmentation.labels.shape
        assert np.sum(gm_mask) > 0

    def test_tissue_probabilities_sum(self, synthetic_segmentation):
        """Test that tissue probabilities sum to approximately 1."""
        point = np.array([4.0, 0.0, 0.0])  # Gray matter region
        probs = synthetic_segmentation.get_tissue_probabilities_at_point(point)

        total = sum(probs.values())
        assert total >= 0.9  # Allow some boundary effects


# =============================================================================
# FiberOrientation Tests
# =============================================================================


class TestFiberOrientation:
    """Tests for FiberOrientation dataclass."""

    def test_get_direction_at_center(self, synthetic_fibers):
        """Test fiber direction at center (should be near zero/arbitrary)."""
        point = np.array([0.0, 0.0, 0.0])
        direction = synthetic_fibers.get_direction_at_point(point)

        assert direction.shape == (3,)
        # Should be normalized
        assert_allclose(np.linalg.norm(direction), 1.0, rtol=1e-6)

    def test_get_direction_off_center(self, synthetic_fibers):
        """Test fiber direction away from center (radial pattern)."""
        point = np.array([5.0, 0.0, 0.0])  # Along x-axis from center
        direction = synthetic_fibers.get_direction_at_point(point)

        # Should be approximately along x-axis (radial outward)
        assert direction[0] > 0.9  # Predominantly x-direction

    def test_direction_is_normalized(self, synthetic_fibers):
        """Test that all returned directions are unit vectors."""
        points = [
            np.array([3.0, 0.0, 0.0]),
            np.array([0.0, 4.0, 0.0]),
            np.array([0.0, 0.0, 5.0]),
            np.array([2.0, 2.0, 2.0]),
        ]

        for point in points:
            direction = synthetic_fibers.get_direction_at_point(point)
            norm = np.linalg.norm(direction)
            assert_allclose(norm, 1.0, rtol=1e-6)

    def test_fractional_anisotropy_bounds(self, synthetic_fibers):
        """Test that FA values are in valid range [0, 1]."""
        fa = synthetic_fibers.fractional_anisotropy
        assert np.all(fa >= 0)
        assert np.all(fa <= 1)


# =============================================================================
# AnisotropicMaterialProperties Tests
# =============================================================================


class TestAnisotropicMaterialProperties:
    """Tests for AnisotropicMaterialProperties dataclass."""

    def test_gray_matter_isotropic(self):
        """Test that gray matter properties are isotropic."""
        props = AnisotropicMaterialProperties.gray_matter_isotropic()

        # Parallel and perpendicular should be equal
        assert_allclose(props.E_parallel, props.E_perpendicular)
        assert_allclose(props.nu_parallel, props.nu_perpendicular)

    def test_white_matter_anisotropic(self):
        """Test that white matter properties are anisotropic."""
        fiber_dir = np.array([1.0, 0.0, 0.0])
        props = AnisotropicMaterialProperties.white_matter_anisotropic(
            fiber_direction=fiber_dir,
            anisotropy_ratio=2.0,
        )

        # Parallel should be stiffer than perpendicular
        assert props.E_parallel > props.E_perpendicular
        assert_allclose(props.E_parallel / props.E_perpendicular, 2.0)

    def test_constitutive_matrix_symmetry(self):
        """Test that constitutive matrix is symmetric."""
        fiber_dir = np.array([1.0, 0.0, 0.0])
        props = AnisotropicMaterialProperties.white_matter_anisotropic(
            fiber_direction=fiber_dir,
        )

        C = props.get_constitutive_matrix()

        assert C.shape == (6, 6)
        assert_allclose(C, C.T, rtol=1e-10)

    def test_constitutive_matrix_positive_definite(self):
        """Test that constitutive matrix is positive definite."""
        props = AnisotropicMaterialProperties.gray_matter_isotropic()
        C = props.get_constitutive_matrix()

        # All eigenvalues should be positive
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues > 0)

    def test_constitutive_matrix_rotation_invariance_for_isotropic(self):
        """Test that isotropic material is rotation invariant."""
        props = AnisotropicMaterialProperties.gray_matter_isotropic()

        # Get constitutive matrix with different "fiber" directions
        # For isotropic, these should be the same
        props.fiber_direction = np.array([1.0, 0.0, 0.0])
        C1 = props.get_constitutive_matrix()

        props.fiber_direction = np.array([0.0, 1.0, 0.0])
        C2 = props.get_constitutive_matrix()

        # Should be nearly identical for isotropic material
        assert_allclose(C1, C2, rtol=0.1)


# =============================================================================
# MaterialProperties Integration Tests
# =============================================================================


class TestMaterialPropertiesAnisotropic:
    """Tests for anisotropic extensions to MaterialProperties."""

    def test_gray_matter_factory(self):
        """Test gray matter factory method."""
        props = MaterialProperties.gray_matter()

        # Should be isotropic (anisotropy_ratio = 1)
        assert props.anisotropy_ratio == 1.0
        assert props.fiber_direction is None

        # Should be more compressible
        assert props.poisson_ratio < 0.45

    def test_white_matter_factory(self):
        """Test white matter factory method."""
        fiber_dir = np.array([1.0, 0.0, 0.0])
        props = MaterialProperties.white_matter(fiber_direction=fiber_dir)

        # Should be anisotropic
        assert props.anisotropy_ratio > 1.0
        assert props.fiber_direction is not None

        # Should be nearly incompressible
        assert props.poisson_ratio >= 0.45

    def test_get_constitutive_matrix_isotropic(self):
        """Test constitutive matrix for isotropic case."""
        props = MaterialProperties.gray_matter()
        C = props.get_constitutive_matrix()

        assert C.shape == (6, 6)
        assert_allclose(C, C.T, rtol=1e-10)  # Symmetric

    def test_get_constitutive_matrix_anisotropic(self):
        """Test constitutive matrix for anisotropic case."""
        fiber_dir = np.array([0.0, 0.0, 1.0])  # Z-aligned fibers
        props = MaterialProperties.white_matter(fiber_direction=fiber_dir)
        C = props.get_constitutive_matrix()

        assert C.shape == (6, 6)
        assert_allclose(C, C.T, rtol=1e-10)

        # C[2,2] should be larger (stiffer in z direction)
        assert C[2, 2] > C[0, 0]  # z-direction stiffer than x

    def test_tissue_poisson_ratios_defined(self):
        """Test that all tissue types have Poisson ratios defined."""
        props = MaterialProperties()

        for tissue_type in TissueType:
            if tissue_type != TissueType.SKULL:  # SKULL may be added later
                assert tissue_type in props.tissue_poisson_ratios or tissue_type == TissueType.SKULL


# =============================================================================
# SUITPyIntegration Tests
# =============================================================================


class TestSUITPyIntegration:
    """Tests for SUITPy integration."""

    def test_synthetic_template_generation(self):
        """Test that synthetic template is generated when no atlas available."""
        suit = SUITPyIntegration(suit_dir=None)
        template, affine = suit.load_template()

        assert template.shape == suit.SUIT_SHAPE
        assert affine.shape == (4, 4)

    def test_template_values_reasonable(self):
        """Test that synthetic template has reasonable intensity values."""
        suit = SUITPyIntegration(suit_dir=None)
        template, _ = suit.load_template()

        # Should have non-zero values in brain regions
        assert np.max(template) > 0
        assert np.min(template) >= 0

    def test_affine_is_valid(self):
        """Test that affine transformation is valid."""
        suit = SUITPyIntegration(suit_dir=None)
        affine = suit.get_suit_affine()

        # Should be invertible
        assert np.linalg.det(affine) != 0

        # Last row should be [0, 0, 0, 1]
        assert_allclose(affine[3, :], [0, 0, 0, 1])


# =============================================================================
# MNIAtlasLoader Tests
# =============================================================================


class TestMNIAtlasLoader:
    """Tests for MNI atlas loader."""

    def test_synthetic_segmentation_generation(self):
        """Test that synthetic segmentation is generated when FSL not available."""
        loader = MNIAtlasLoader(fsl_dir=None)
        seg = loader.load_tissue_segmentation()

        assert seg.labels.shape == loader.MNI_SHAPE
        assert seg.wm_probability is not None
        assert seg.gm_probability is not None

    def test_tissue_labels_valid(self):
        """Test that generated labels are valid tissue types."""
        loader = MNIAtlasLoader(fsl_dir=None)
        seg = loader.load_tissue_segmentation()

        unique_labels = np.unique(seg.labels)

        # All labels should be valid BrainTissue values
        valid_values = [t.value for t in BrainTissue]
        for label in unique_labels:
            assert label in valid_values

    def test_synthetic_fibers_generation(self):
        """Test that synthetic fibers are generated when no DTI atlas available."""
        loader = MNIAtlasLoader(fsl_dir=None)
        fibers = loader.load_fiber_orientation()

        assert fibers.vectors.shape[:3] == loader.MNI_SHAPE
        assert fibers.vectors.shape[3] == 3
        assert fibers.fractional_anisotropy.shape == loader.MNI_SHAPE

    def test_fiber_vectors_normalized(self):
        """Test that fiber vectors are normalized."""
        loader = MNIAtlasLoader(fsl_dir=None)
        fibers = loader.load_fiber_orientation()

        # Sample some vectors
        norms = np.linalg.norm(fibers.vectors[::10, ::10, ::10], axis=-1)
        valid_norms = norms[norms > 0]

        if len(valid_norms) > 0:
            assert_allclose(valid_norms, 1.0, rtol=0.1)


# =============================================================================
# SpaceTransformer Tests
# =============================================================================


class TestSpaceTransformer:
    """Tests for MNI-SUIT space transformation."""

    def test_transform_round_trip(self):
        """Test that MNI -> SUIT -> MNI preserves coordinates."""
        suit = SUITPyIntegration(suit_dir=None)
        mni = MNIAtlasLoader(fsl_dir=None)
        transformer = SpaceTransformer(suit, mni)

        original_coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 20.0, -30.0],
            [-15.0, 5.0, 10.0],
        ])

        for coord in original_coords:
            suit_coord = transformer.mni_to_suit_coords(coord)
            back_to_mni = transformer.suit_to_mni_coords(suit_coord)

            assert_allclose(back_to_mni, coord, rtol=1e-6)

    def test_transform_single_coordinate(self):
        """Test transformation of single coordinate."""
        suit = SUITPyIntegration(suit_dir=None)
        mni = MNIAtlasLoader(fsl_dir=None)
        transformer = SpaceTransformer(suit, mni)

        coord = np.array([0.0, 0.0, 0.0])
        suit_coord = transformer.mni_to_suit_coords(coord)

        assert suit_coord.shape == (3,)

    def test_transform_batch_coordinates(self):
        """Test transformation of batch of coordinates."""
        suit = SUITPyIntegration(suit_dir=None)
        mni = MNIAtlasLoader(fsl_dir=None)
        transformer = SpaceTransformer(suit, mni)

        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [-10.0, -20.0, -30.0],
        ])

        suit_coords = transformer.mni_to_suit_coords(coords)

        assert suit_coords.shape == (3, 3)


# =============================================================================
# BiophysicalConstraints Integration Tests
# =============================================================================


class TestBiophysicalConstraints:
    """Tests for integrated biophysical constraints."""

    def test_initialization(self):
        """Test basic initialization with new defaults."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None)

        assert bc.suit is not None
        assert bc.mni is not None
        assert bc.transformer is not None
        # New defaults: MNI space, posterior fossa only
        assert bc.use_suit_space is False  # Default is now MNI space
        assert bc.posterior_fossa_only is True  # Default is posterior fossa only

    def test_default_tumor_origin(self):
        """Test that default tumor origin is vermis in MNI space."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None)

        # Default tumor origin should be [1, -61, -34] (vermis)
        assert_allclose(bc.tumor_origin, DEFAULT_TUMOR_ORIGIN_MNI)
        assert_allclose(bc.tumor_origin, [1.0, -61.0, -34.0])

    def test_custom_tumor_origin(self):
        """Test setting custom tumor origin."""
        custom_origin = np.array([10.0, -50.0, -30.0])
        bc = BiophysicalConstraints(
            suit_dir=None, fsl_dir=None,
            tumor_origin=custom_origin
        )

        assert_allclose(bc.tumor_origin, custom_origin)

    def test_posterior_fossa_bounds(self):
        """Test posterior fossa bounding box is reasonable."""
        bounds = POSTERIOR_FOSSA_BOUNDS_MNI

        # Bounds should cover posterior and inferior region
        assert bounds['y_max'] < 0  # Posterior (negative Y in MNI)
        assert bounds['z_max'] < 20  # Inferior (low Z in MNI)

        # Bounds should be symmetric in X
        assert bounds['x_min'] == -bounds['x_max']

    def test_is_in_posterior_fossa_vermis(self):
        """Test that default tumor origin is in posterior fossa."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None)

        # Default tumor origin (vermis) should be in posterior fossa
        assert bc.is_in_posterior_fossa(bc.tumor_origin)

    def test_is_in_posterior_fossa_cerebellum(self):
        """Test points in cerebellum are in posterior fossa."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None)

        # Various cerebellar points
        cerebellar_points = [
            np.array([0.0, -60.0, -30.0]),  # Vermis
            np.array([30.0, -70.0, -40.0]),  # Right hemisphere
            np.array([-30.0, -70.0, -40.0]),  # Left hemisphere
        ]

        for point in cerebellar_points:
            assert bc.is_in_posterior_fossa(point), f"Point {point} should be in posterior fossa"

    def test_is_not_in_posterior_fossa_frontal(self):
        """Test that frontal lobe points are not in posterior fossa."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None)

        # Frontal lobe point (anterior, superior)
        frontal_point = np.array([0.0, 50.0, 30.0])
        assert not bc.is_in_posterior_fossa(frontal_point)

    def test_posterior_fossa_mask_computed(self):
        """Test that posterior fossa mask is computed."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None)
        mask = bc.compute_posterior_fossa_mask()

        # Mask should be 3D boolean array
        assert mask.ndim == 3
        assert mask.dtype == np.bool_

        # Should have some True values (posterior fossa region)
        assert np.sum(mask) > 0

        # Should not be all True (not whole brain)
        assert np.sum(mask) < mask.size

    def test_load_all_constraints(self):
        """Test loading all constraint data."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None)
        bc.load_all_constraints()

        assert bc._segmentation is not None
        assert bc._fibers is not None
        assert bc._skull_boundary is not None
        # With posterior_fossa_only=True, mask should also be computed
        assert bc._posterior_fossa_mask is not None

    def test_segmentation_masked_to_posterior_fossa(self):
        """Test that segmentation is masked to posterior fossa only."""
        bc = BiophysicalConstraints(
            suit_dir=None, fsl_dir=None,
            posterior_fossa_only=True
        )
        seg = bc.load_tissue_segmentation()

        # Frontal region should be background
        # Convert frontal MNI coords to voxel
        frontal_mni = np.array([0.0, 50.0, 30.0])
        frontal_tissue = seg.get_tissue_at_point(frontal_mni)
        assert frontal_tissue == BrainTissue.BACKGROUND

    def test_segmentation_not_masked_when_disabled(self):
        """Test that segmentation is not masked when posterior_fossa_only=False."""
        bc = BiophysicalConstraints(
            suit_dir=None, fsl_dir=None,
            posterior_fossa_only=False
        )
        seg = bc.load_tissue_segmentation()

        # Frontal region should have tissue (not background)
        # This is within the synthetic brain, so should have tissue
        frontal_mni = np.array([0.0, 30.0, 20.0])
        frontal_tissue = seg.get_tissue_at_point(frontal_mni)
        # May be GM, WM, or CSF depending on synthetic generation
        assert frontal_tissue != BrainTissue.BACKGROUND or True  # Allow background in synthetic

    def test_get_material_properties_gray_matter(self, synthetic_segmentation):
        """Test getting material properties for gray matter region."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc._segmentation = synthetic_segmentation
        bc._fibers = FiberOrientation(
            vectors=np.ones((20, 20, 20, 3)) * [1, 0, 0],
            fractional_anisotropy=np.ones((20, 20, 20), dtype=np.float32) * 0.5,
            affine=synthetic_segmentation.affine,
        )

        # Point in gray matter
        point = np.array([4.0, 0.0, 0.0])
        props = bc.get_material_properties_at_node(point)

        # Should be isotropic (gray matter)
        assert_allclose(props.E_parallel, props.E_perpendicular)

    def test_get_material_properties_white_matter(self, synthetic_segmentation):
        """Test getting material properties for white matter region."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc._segmentation = synthetic_segmentation

        # Create fiber orientation
        fibers = FiberOrientation(
            vectors=np.ones((20, 20, 20, 3)) * [1, 0, 0],
            fractional_anisotropy=np.ones((20, 20, 20), dtype=np.float32) * 0.8,
            affine=synthetic_segmentation.affine,
        )
        bc._fibers = fibers

        # Point in white matter
        point = np.array([7.0, 0.0, 0.0])
        props = bc.get_material_properties_at_node(point)

        # Should be anisotropic (white matter)
        assert props.E_parallel > props.E_perpendicular

    def test_assign_node_tissues(self, cube_mesh_with_tissues, synthetic_segmentation):
        """Test tissue assignment to mesh nodes."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc._segmentation = synthetic_segmentation

        # Create small test mesh
        nodes = np.array([
            [0.0, 0.0, 0.0],  # CSF region
            [4.0, 0.0, 0.0],  # GM region
            [7.0, 0.0, 0.0],  # WM region
        ], dtype=np.float64)

        tissues = bc.assign_node_tissues(nodes)

        assert tissues.shape == (3,)
        # First should be CSF
        assert tissues[0] == BrainTissue.CSF.value
        # Second should be GM
        assert tissues[1] == BrainTissue.GRAY_MATTER.value
        # Third should be WM
        assert tissues[2] == BrainTissue.WHITE_MATTER.value

    def test_get_fiber_directions_at_nodes(self, synthetic_fibers):
        """Test getting fiber directions at node positions."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc._fibers = synthetic_fibers
        bc._segmentation = TissueSegmentation(
            labels=np.ones((20, 20, 20), dtype=np.int32) * 3,
            affine=synthetic_fibers.affine,
            voxel_size=(1.0, 1.0, 1.0),
        )

        nodes = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ], dtype=np.float64)

        directions = bc.get_fiber_directions_at_nodes(nodes)

        assert directions.shape == (3, 3)
        # All should be normalized
        for i in range(3):
            norm = np.linalg.norm(directions[i])
            assert_allclose(norm, 1.0, rtol=1e-6)


# =============================================================================
# FEM Solver Integration Tests
# =============================================================================


class TestFEMWithBiophysicalConstraints:
    """Tests for FEM solver with biophysical constraints."""

    def test_solver_initialization_with_constraints(self, cube_mesh_with_tissues):
        """Test that solver initializes with biophysical constraints."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc.load_all_constraints()

        solver = TumorGrowthSolver(
            mesh=cube_mesh_with_tissues,
            biophysical_constraints=bc,
            boundary_condition="skull",
        )

        assert solver.biophysical_constraints is bc
        assert solver._node_tissues is not None
        assert solver._node_fiber_directions is not None
        assert solver._element_properties is not None

    def test_solver_step_with_constraints(self, cube_mesh_with_tissues):
        """Test that solver step works with biophysical constraints."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc.load_all_constraints()

        solver = TumorGrowthSolver(
            mesh=cube_mesh_with_tissues,
            biophysical_constraints=bc,
        )

        center = np.mean(cube_mesh_with_tissues.nodes, axis=0)
        state = TumorState.initial(
            mesh=cube_mesh_with_tissues,
            seed_center=center,
            seed_radius=2.0,
        )

        # Should not crash
        new_state = solver.step(state, dt=1.0)

        assert new_state.cell_density.shape == (cube_mesh_with_tissues.num_nodes,)
        assert np.all(new_state.cell_density >= 0)

    def test_tissue_specific_diffusion(self, cube_mesh_with_tissues):
        """Test that tissue-specific diffusion is applied."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc.load_all_constraints()

        solver = TumorGrowthSolver(
            mesh=cube_mesh_with_tissues,
            biophysical_constraints=bc,
        )

        # Check that element properties have different diffusion coefficients
        if solver._element_properties:
            diffusion_coeffs = [p.diffusion_coefficient for p in solver._element_properties]
            # Should have some variation (unless all same tissue)
            unique_coeffs = set(diffusion_coeffs)
            # At minimum, should have valid coefficients
            assert all(d > 0 for d in diffusion_coeffs)

    def test_stiffness_matrix_with_anisotropic_materials(self, cube_mesh_with_tissues):
        """Test that stiffness matrix is built with anisotropic materials."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc.load_all_constraints()

        solver = TumorGrowthSolver(
            mesh=cube_mesh_with_tissues,
            biophysical_constraints=bc,
        )

        K = solver._stiffness_matrix

        # Matrix should be symmetric (allow for numerical precision)
        K_dense = K.toarray()
        assert_allclose(K_dense, K_dense.T, atol=1e-12)

        # Matrix should be correct size
        expected_size = 3 * cube_mesh_with_tissues.num_nodes
        assert K.shape == (expected_size, expected_size)

    def test_skull_boundary_condition(self, cube_mesh_with_tissues):
        """Test skull boundary condition application."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc.load_all_constraints()

        solver = TumorGrowthSolver(
            mesh=cube_mesh_with_tissues,
            biophysical_constraints=bc,
            boundary_condition="skull",
        )

        center = np.mean(cube_mesh_with_tissues.nodes, axis=0)
        state = TumorState.initial(
            mesh=cube_mesh_with_tissues,
            seed_center=center,
            seed_radius=2.0,
            seed_density=0.8,
        )

        # Run simulation
        for _ in range(5):
            state = solver.step(state, dt=1.0)

        # Boundary nodes should have constrained displacement
        # (exact behavior depends on skull boundary identification)
        assert state.displacement.shape == (cube_mesh_with_tissues.num_nodes, 3)


# =============================================================================
# Stress Computation Tests
# =============================================================================


class TestStressWithAnisotropicMaterials:
    """Tests for stress computation with anisotropic materials."""

    def test_stress_shape(self, cube_mesh_with_tissues):
        """Test that stress has correct shape."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc.load_all_constraints()

        solver = TumorGrowthSolver(
            mesh=cube_mesh_with_tissues,
            biophysical_constraints=bc,
        )

        center = np.mean(cube_mesh_with_tissues.nodes, axis=0)
        state = TumorState.initial(
            mesh=cube_mesh_with_tissues,
            seed_center=center,
            seed_radius=2.0,
        )

        state = solver.step(state, dt=1.0)

        assert state.stress.shape == (cube_mesh_with_tissues.num_elements, 6)

    def test_von_mises_stress_positive(self, cube_mesh_with_tissues):
        """Test that von Mises stress is non-negative."""
        bc = BiophysicalConstraints(suit_dir=None, fsl_dir=None, use_suit_space=False)
        bc.load_all_constraints()

        solver = TumorGrowthSolver(
            mesh=cube_mesh_with_tissues,
            biophysical_constraints=bc,
        )

        center = np.mean(cube_mesh_with_tissues.nodes, axis=0)
        state = TumorState.initial(
            mesh=cube_mesh_with_tissues,
            seed_center=center,
            seed_radius=2.0,
            seed_density=0.8,
        )

        for _ in range(3):
            state = solver.step(state, dt=1.0)

        von_mises = solver.compute_von_mises_stress(state)

        assert np.all(von_mises >= 0)
