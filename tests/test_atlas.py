"""
Tests for SUIT atlas loading and processing module.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pft_fem.atlas import SUITAtlasLoader, AtlasProcessor, AtlasData, AtlasRegion


# =============================================================================
# SUITAtlasLoader Tests
# =============================================================================


class TestSUITAtlasLoader:
    """Tests for SUITAtlasLoader."""

    def test_loader_initialization_no_path(self):
        """Test loader initialization without atlas path."""
        loader = SUITAtlasLoader()
        assert loader.atlas_dir is None
        assert loader._cached_data is None

    def test_load_synthetic(self):
        """Test loading synthetic atlas data."""
        loader = SUITAtlasLoader()
        data = loader.load()

        assert isinstance(data, AtlasData)
        assert data.template is not None
        assert data.labels is not None
        assert data.affine is not None

    def test_load_returns_atlas_data(self, synthetic_atlas):
        """Test that load returns AtlasData instance."""
        assert isinstance(synthetic_atlas, AtlasData)

    def test_atlas_data_shapes(self, synthetic_atlas):
        """Test that atlas data has correct shapes."""
        assert synthetic_atlas.template.ndim == 3
        assert synthetic_atlas.labels.ndim == 3
        assert synthetic_atlas.affine.shape == (4, 4)

        # Template and labels should have same shape
        assert synthetic_atlas.template.shape == synthetic_atlas.labels.shape
        assert synthetic_atlas.shape == synthetic_atlas.template.shape

    def test_atlas_data_types(self, synthetic_atlas):
        """Test that atlas data has correct dtypes."""
        assert synthetic_atlas.template.dtype == np.float32
        assert synthetic_atlas.labels.dtype == np.int32
        assert synthetic_atlas.affine.dtype == np.float64

    def test_voxel_size_positive(self, synthetic_atlas):
        """Test that voxel sizes are positive."""
        for v in synthetic_atlas.voxel_size:
            assert v > 0

    def test_regions_extracted(self, synthetic_atlas):
        """Test that regions are extracted from labels."""
        assert len(synthetic_atlas.regions) > 0

        for label_id, region in synthetic_atlas.regions.items():
            assert isinstance(region, AtlasRegion)
            assert region.label_id == label_id
            assert region.volume_mm3 > 0

    def test_load_caching(self):
        """Test that caching works correctly."""
        loader = SUITAtlasLoader()

        data1 = loader.load(use_cache=True)
        data2 = loader.load(use_cache=True)

        # Should be same object
        assert data1 is data2

    def test_load_no_cache(self):
        """Test loading without cache."""
        loader = SUITAtlasLoader()

        data1 = loader.load(use_cache=False)
        data2 = loader.load(use_cache=False)

        # Should be different objects (regenerated)
        assert data1 is not data2

    def test_region_labels_defined(self):
        """Test that region labels are defined."""
        assert len(SUITAtlasLoader.REGION_LABELS) > 0

        # Key regions should be defined
        assert 29 in SUITAtlasLoader.REGION_LABELS  # Brainstem
        assert 30 in SUITAtlasLoader.REGION_LABELS  # Fourth Ventricle

    def test_get_region_by_name(self):
        """Test finding region by name."""
        loader = SUITAtlasLoader()

        # Test exact partial match
        brainstem_id = loader.get_region_by_name("Brainstem")
        assert brainstem_id == 29

        # Test case insensitive
        ventricle_id = loader.get_region_by_name("ventricle")
        assert ventricle_id == 30

        # Test partial match
        crus_id = loader.get_region_by_name("Crus I")
        assert crus_id is not None

        # Test non-existent
        none_id = loader.get_region_by_name("NonexistentRegion")
        assert none_id is None


# =============================================================================
# AtlasRegion Tests
# =============================================================================


class TestAtlasRegion:
    """Tests for AtlasRegion dataclass."""

    def test_region_in_atlas(self, synthetic_atlas):
        """Test that regions have valid data."""
        for region in synthetic_atlas.regions.values():
            # Mask should match label
            expected_count = np.sum(synthetic_atlas.labels == region.label_id)
            actual_count = np.sum(region.mask)
            assert expected_count == actual_count

            # Volume should be positive
            assert region.volume_mm3 > 0

            # Centroid should be within volume bounds
            shape = synthetic_atlas.shape
            assert 0 <= region.centroid[0] < shape[0]
            assert 0 <= region.centroid[1] < shape[1]
            assert 0 <= region.centroid[2] < shape[2]


# =============================================================================
# AtlasProcessor Tests
# =============================================================================


class TestAtlasProcessor:
    """Tests for AtlasProcessor."""

    def test_processor_initialization(self, synthetic_atlas):
        """Test processor initialization."""
        processor = AtlasProcessor(synthetic_atlas)
        assert processor.atlas is synthetic_atlas

    def test_get_tissue_mask_cerebellum(self, synthetic_atlas):
        """Test getting cerebellum mask."""
        processor = AtlasProcessor(synthetic_atlas)
        mask = processor.get_tissue_mask("cerebellum")

        assert mask.dtype == bool
        assert mask.shape == synthetic_atlas.shape
        assert np.any(mask)  # Should have some True values

    def test_get_tissue_mask_brainstem(self, synthetic_atlas):
        """Test getting brainstem mask."""
        processor = AtlasProcessor(synthetic_atlas)
        mask = processor.get_tissue_mask("brainstem")

        assert mask.dtype == bool
        assert mask.shape == synthetic_atlas.shape

    def test_get_tissue_mask_ventricle(self, synthetic_atlas):
        """Test getting ventricle mask."""
        processor = AtlasProcessor(synthetic_atlas)
        mask = processor.get_tissue_mask("ventricle")

        assert mask.dtype == bool
        assert mask.shape == synthetic_atlas.shape

    def test_get_tissue_mask_all(self, synthetic_atlas):
        """Test getting all tissue mask."""
        processor = AtlasProcessor(synthetic_atlas)
        mask = processor.get_tissue_mask("all")

        # Should include all non-background voxels
        expected = synthetic_atlas.labels > 0
        assert np.array_equal(mask, expected)

    def test_get_tissue_mask_invalid(self, synthetic_atlas):
        """Test that invalid tissue type raises error."""
        processor = AtlasProcessor(synthetic_atlas)

        with pytest.raises(ValueError):
            processor.get_tissue_mask("invalid_tissue")

    def test_compute_distance_field_signed(self, synthetic_atlas):
        """Test signed distance field computation."""
        processor = AtlasProcessor(synthetic_atlas)
        dist = processor.compute_distance_field(signed=True)

        assert dist.dtype == np.float32
        assert dist.shape == synthetic_atlas.shape

        # Inside should be negative, outside positive
        mask = processor.get_tissue_mask("all")
        assert np.any(dist[mask] < 0)
        assert np.any(dist[~mask] > 0)

    def test_compute_distance_field_unsigned(self, synthetic_atlas):
        """Test unsigned distance field computation."""
        processor = AtlasProcessor(synthetic_atlas)
        dist = processor.compute_distance_field(signed=False)

        # All values should be non-negative
        assert np.all(dist >= 0)

        # Inside mask should have zero distance
        mask = processor.get_tissue_mask("all")
        assert_allclose(dist[mask], 0.0)

    def test_extract_surface_points(self, synthetic_atlas):
        """Test surface point extraction."""
        processor = AtlasProcessor(synthetic_atlas)
        points = processor.extract_surface_points()

        assert points.ndim == 2
        assert points.shape[1] == 3
        assert len(points) > 0

    def test_extract_surface_points_spacing(self, synthetic_atlas):
        """Test surface point extraction with spacing."""
        processor = AtlasProcessor(synthetic_atlas)

        points_dense = processor.extract_surface_points(spacing=1)
        points_sparse = processor.extract_surface_points(spacing=3)

        # Sparse should have fewer points
        assert len(points_sparse) < len(points_dense)

    def test_resample(self, synthetic_atlas):
        """Test atlas resampling."""
        processor = AtlasProcessor(synthetic_atlas)

        # Downsample
        target_shape = tuple(s // 2 for s in synthetic_atlas.shape)
        resampled = processor.resample(target_shape)

        assert isinstance(resampled, AtlasData)
        assert resampled.shape == target_shape
        assert resampled.template.shape == target_shape
        assert resampled.labels.shape == target_shape

    def test_resample_preserves_structure(self, synthetic_atlas):
        """Test that resampling preserves tissue structure."""
        processor = AtlasProcessor(synthetic_atlas)

        # Downsample and upsample
        small_shape = tuple(s // 2 for s in synthetic_atlas.shape)
        resampled = processor.resample(small_shape)

        # Should still have regions
        assert len(resampled.regions) > 0

    def test_get_bounding_box(self, synthetic_atlas):
        """Test bounding box computation."""
        processor = AtlasProcessor(synthetic_atlas)
        min_coords, max_coords = processor.get_bounding_box()

        # Min should be less than max
        for i in range(3):
            assert min_coords[i] < max_coords[i]

        # Should be within volume bounds
        for i in range(3):
            assert min_coords[i] >= 0
            assert max_coords[i] <= synthetic_atlas.shape[i]

    def test_get_bounding_box_with_padding(self, synthetic_atlas):
        """Test bounding box with padding."""
        processor = AtlasProcessor(synthetic_atlas)

        min_no_pad, max_no_pad = processor.get_bounding_box(padding=0)
        min_pad, max_pad = processor.get_bounding_box(padding=5)

        # Padded box should be larger
        for i in range(3):
            assert min_pad[i] <= min_no_pad[i]
            assert max_pad[i] >= max_no_pad[i]

    def test_crop_to_region(self, synthetic_atlas):
        """Test cropping to tissue region."""
        processor = AtlasProcessor(synthetic_atlas)
        cropped = processor.crop_to_region(padding=2)

        assert isinstance(cropped, AtlasData)

        # Cropped should be smaller or equal
        for i in range(3):
            assert cropped.shape[i] <= synthetic_atlas.shape[i]

        # Should still contain all tissue
        original_tissue_count = np.sum(synthetic_atlas.labels > 0)
        cropped_tissue_count = np.sum(cropped.labels > 0)

        # Account for padding at edges
        assert cropped_tissue_count >= original_tissue_count * 0.99


# =============================================================================
# Synthetic Atlas Quality Tests
# =============================================================================


class TestSyntheticAtlasQuality:
    """Tests for quality of synthetic atlas generation."""

    def test_synthetic_has_multiple_regions(self, synthetic_atlas):
        """Test that synthetic atlas has multiple anatomical regions."""
        unique_labels = np.unique(synthetic_atlas.labels)
        # Should have background (0) plus at least a few regions
        assert len(unique_labels) >= 4

    def test_synthetic_regions_separated(self, synthetic_atlas):
        """Test that major regions are spatially separated."""
        processor = AtlasProcessor(synthetic_atlas)

        cerebellum = processor.get_tissue_mask("cerebellum")
        brainstem = processor.get_tissue_mask("brainstem")
        ventricle = processor.get_tissue_mask("ventricle")

        # Regions should not completely overlap
        # (Some overlap at boundaries is OK)
        assert not np.array_equal(cerebellum, brainstem)
        assert not np.array_equal(cerebellum, ventricle)

    def test_synthetic_template_intensity_range(self, synthetic_atlas):
        """Test that template intensities are in reasonable range."""
        template = synthetic_atlas.template

        # Should be non-negative
        assert np.all(template >= 0)

        # Should have some variation
        assert np.std(template) > 0

        # Non-background regions should have higher intensity
        mask = synthetic_atlas.labels > 0
        bg_mean = np.mean(template[~mask])
        tissue_mean = np.mean(template[mask])

        assert tissue_mean > bg_mean

    def test_synthetic_affine_valid(self, synthetic_atlas):
        """Test that affine transformation is valid."""
        affine = synthetic_atlas.affine

        # Should be invertible
        det = np.linalg.det(affine)
        assert det != 0

        # Diagonal should encode voxel sizes (for simple affines)
        for i in range(3):
            assert abs(affine[i, i]) > 0

    def test_synthetic_reproducible(self):
        """Test that synthetic generation is somewhat consistent."""
        loader = SUITAtlasLoader()

        # Same shape/voxel size should give similar structure
        data1 = loader._generate_synthetic_atlas(shape=(50, 50, 50))
        data2 = loader._generate_synthetic_atlas(shape=(50, 50, 50))

        # Shapes should match
        assert data1.shape == data2.shape

        # Same regions should exist
        labels1 = set(np.unique(data1.labels))
        labels2 = set(np.unique(data2.labels))

        assert labels1 == labels2


# =============================================================================
# Integration Tests
# =============================================================================


class TestAtlasIntegration:
    """Integration tests for atlas module."""

    def test_full_atlas_workflow(self):
        """Test complete atlas loading and processing workflow."""
        # Load atlas
        loader = SUITAtlasLoader()
        atlas = loader.load()

        # Process atlas
        processor = AtlasProcessor(atlas)

        # Get masks
        cerebellum_mask = processor.get_tissue_mask("cerebellum")
        all_mask = processor.get_tissue_mask("all")

        # Compute distance field
        dist_field = processor.compute_distance_field()

        # Extract surface
        surface_points = processor.extract_surface_points()

        # Get bounding box
        bbox = processor.get_bounding_box()

        # Crop
        cropped = processor.crop_to_region()

        # Verify all operations completed successfully
        assert cerebellum_mask is not None
        assert all_mask is not None
        assert dist_field is not None
        assert len(surface_points) > 0
        assert len(bbox) == 2
        assert cropped is not None

    def test_atlas_for_mesh_generation(self, synthetic_atlas):
        """Test that atlas can be used for mesh generation."""
        from pft_fem.mesh import MeshGenerator

        processor = AtlasProcessor(synthetic_atlas)
        mask = processor.get_tissue_mask("all")

        generator = MeshGenerator()
        mesh = generator.from_mask(
            mask=mask,
            voxel_size=synthetic_atlas.voxel_size,
            labels=synthetic_atlas.labels,
            simplify=False,
        )

        # Should produce valid mesh
        assert mesh.num_nodes > 0
        assert mesh.num_elements > 0
