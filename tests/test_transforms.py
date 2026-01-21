"""
Tests for spatial transform tracking and ANTsPy-compatible export.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

from pft_fem.transforms import (
    SpatialTransform,
    ANTsTransformExporter,
    compute_transform_from_simulation,
)


class TestSpatialTransform:
    """Tests for the SpatialTransform class."""

    def test_identity_transform(self):
        """Test identity transform creation."""
        shape = (10, 10, 10)
        voxel_size = (2.0, 2.0, 2.0)

        transform = SpatialTransform.identity(shape, voxel_size)

        assert transform.reference_shape == shape
        assert transform.voxel_size == voxel_size
        assert transform.displacement_field.shape == (*shape, 3)
        assert np.allclose(transform.displacement_field, 0)
        assert transform.source_space == "SUIT"
        assert transform.target_space == "deformed"

    def test_identity_transform_default_affine(self):
        """Test identity transform with default affine."""
        shape = (10, 10, 10)
        voxel_size = (1.5, 1.5, 1.5)

        transform = SpatialTransform.identity(shape, voxel_size)

        expected_affine = np.diag([1.5, 1.5, 1.5, 1.0])
        assert np.allclose(transform.affine, expected_affine)

    def test_identity_transform_custom_affine(self):
        """Test identity transform with custom affine."""
        shape = (10, 10, 10)
        voxel_size = (2.0, 2.0, 2.0)
        custom_affine = np.array([
            [2.0, 0, 0, -10],
            [0, 2.0, 0, -20],
            [0, 0, 2.0, -30],
            [0, 0, 0, 1],
        ])

        transform = SpatialTransform.identity(shape, voxel_size, custom_affine)

        assert np.allclose(transform.affine, custom_affine)

    def test_from_displacement_field(self):
        """Test creating transform from displacement field."""
        shape = (8, 8, 8)
        displacement = np.random.randn(*shape, 3).astype(np.float32)
        affine = np.eye(4) * 2
        affine[3, 3] = 1

        transform = SpatialTransform.from_displacement_field(
            displacement, affine, source_space="MNI", target_space="subject"
        )

        assert np.allclose(transform.displacement_field, displacement)
        assert transform.source_space == "MNI"
        assert transform.target_space == "subject"

    def test_invalid_affine_shape_raises(self):
        """Test that invalid affine shape raises error."""
        shape = (10, 10, 10)
        bad_affine = np.eye(3)  # Should be 4x4
        displacement = np.zeros((*shape, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Affine must be 4x4"):
            SpatialTransform(
                affine=bad_affine,
                displacement_field=displacement,
                reference_shape=shape,
                voxel_size=(1.0, 1.0, 1.0),
            )

    def test_invalid_displacement_shape_raises(self):
        """Test that invalid displacement shape raises error."""
        shape = (10, 10, 10)
        affine = np.eye(4)
        bad_displacement = np.zeros((5, 5, 5, 3), dtype=np.float32)  # Wrong shape

        with pytest.raises(ValueError, match="does not match expected"):
            SpatialTransform(
                affine=affine,
                displacement_field=bad_displacement,
                reference_shape=shape,
                voxel_size=(1.0, 1.0, 1.0),
            )

    def test_get_displacement_at_point_nearest(self):
        """Test displacement lookup with nearest neighbor interpolation."""
        shape = (5, 5, 5)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        displacement[2, 2, 2] = [1.0, 2.0, 3.0]
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        # Query at center point
        disp = transform.get_displacement_at_point(
            np.array([2.0, 2.0, 2.0]), interpolation="nearest"
        )
        assert np.allclose(disp, [1.0, 2.0, 3.0])

        # Query at nearby point (should round to center)
        disp = transform.get_displacement_at_point(
            np.array([2.4, 2.4, 2.4]), interpolation="nearest"
        )
        assert np.allclose(disp, [1.0, 2.0, 3.0])

    def test_get_displacement_at_point_linear(self):
        """Test displacement lookup with linear interpolation."""
        shape = (5, 5, 5)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        displacement[2, 2, 2] = [8.0, 8.0, 8.0]
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        # Query at exact center should give exact value
        disp = transform.get_displacement_at_point(
            np.array([2.0, 2.0, 2.0]), interpolation="linear"
        )
        assert np.allclose(disp, [8.0, 8.0, 8.0])

        # Query at midpoint between zero and center should be interpolated
        disp = transform.get_displacement_at_point(
            np.array([1.5, 1.5, 1.5]), interpolation="linear"
        )
        # Should be some fraction of the displacement
        assert np.all(disp >= 0) and np.all(disp <= 8.0)

    def test_transform_point_forward(self):
        """Test forward point transformation."""
        shape = (10, 10, 10)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        # Set uniform displacement
        displacement[:, :, :] = [1.0, 2.0, 3.0]
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        point = np.array([5.0, 5.0, 5.0])
        transformed = transform.transform_point(point)

        expected = point + np.array([1.0, 2.0, 3.0])
        assert np.allclose(transformed, expected)

    def test_get_displacement_magnitude(self):
        """Test displacement magnitude computation."""
        shape = (5, 5, 5)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        displacement[2, 2, 2] = [3.0, 4.0, 0.0]  # Magnitude = 5.0
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        magnitude = transform.get_displacement_magnitude()
        assert magnitude.shape == shape
        assert np.isclose(magnitude[2, 2, 2], 5.0)

    def test_get_max_displacement(self):
        """Test max displacement computation."""
        shape = (5, 5, 5)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        displacement[1, 1, 1] = [3.0, 0.0, 0.0]  # Magnitude = 3.0
        displacement[2, 2, 2] = [3.0, 4.0, 0.0]  # Magnitude = 5.0
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        max_disp = transform.get_max_displacement()
        assert np.isclose(max_disp, 5.0)

    def test_get_mean_displacement(self):
        """Test mean displacement computation."""
        shape = (5, 5, 5)
        displacement = np.ones((*shape, 3), dtype=np.float32)
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        mean_disp = transform.get_mean_displacement()
        expected = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        assert np.isclose(mean_disp, expected, rtol=0.01)

    def test_get_mean_displacement_with_mask(self):
        """Test mean displacement with mask."""
        shape = (5, 5, 5)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        displacement[0, 0, 0] = [10.0, 0.0, 0.0]
        displacement[2, 2, 2] = [3.0, 4.0, 0.0]  # Magnitude = 5.0
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        # Mask that only includes the second point
        mask = np.zeros(shape, dtype=bool)
        mask[2, 2, 2] = True

        mean_disp = transform.get_mean_displacement(mask)
        assert np.isclose(mean_disp, 5.0)

    def test_jacobian_determinant(self):
        """Test Jacobian determinant computation."""
        shape = (10, 10, 10)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        jacobian = transform.get_jacobian_determinant()
        assert jacobian.shape == shape
        # For zero displacement, Jacobian should be ~1 everywhere
        assert np.allclose(jacobian, 1.0, atol=0.01)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        shape = (5, 5, 5)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        affine = np.eye(4) * 2
        affine[3, 3] = 1

        transform = SpatialTransform.from_displacement_field(
            displacement, affine, source_space="SUIT", target_space="deformed"
        )
        transform.metadata = {"test_key": "test_value"}

        d = transform.to_dict()

        assert d["source_space"] == "SUIT"
        assert d["target_space"] == "deformed"
        assert d["reference_shape"] == list(shape)
        assert d["max_displacement_mm"] == 0.0
        assert d["metadata"]["test_key"] == "test_value"


class TestANTsTransformExporter:
    """Tests for ANTsPy-compatible transform export."""

    def test_export_displacement_field(self):
        """Test exporting displacement field as NIfTI."""
        shape = (10, 10, 10)
        displacement = np.random.randn(*shape, 3).astype(np.float32)
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ANTsTransformExporter(tmpdir)
            path = exporter.export_displacement_field(transform)

            assert path.exists()
            assert path.suffix == ".gz"

            # Verify the file can be loaded
            import nibabel as nib
            img = nib.load(path)
            data = img.get_fdata()

            # ANTs format: (X, Y, Z, 1, 3)
            assert data.shape == (*shape, 1, 3)
            assert np.allclose(data[..., 0, :], displacement)

    def test_export_affine_transform(self):
        """Test exporting affine transform as .mat file."""
        shape = (10, 10, 10)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        affine = np.array([
            [2.0, 0, 0, -10],
            [0, 2.0, 0, -20],
            [0, 0, 2.0, -30],
            [0, 0, 0, 1],
        ])

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ANTsTransformExporter(tmpdir)
            path = exporter.export_affine_transform(transform)

            assert path.exists()
            assert path.suffix == ".mat"

            # Verify content
            content = path.read_text()
            assert "AffineTransform" in content
            assert "Parameters:" in content

    def test_export_composite_transform(self):
        """Test exporting complete composite transform."""
        shape = (8, 8, 8)
        displacement = np.random.randn(*shape, 3).astype(np.float32) * 0.5
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ANTsTransformExporter(tmpdir)
            paths = exporter.export_composite_transform(
                transform, base_filename="test_transform", include_inverse=True
            )

            assert "displacement_field" in paths
            assert "affine" in paths
            assert "inverse_displacement" in paths
            assert "metadata" in paths
            assert "jacobian_determinant" in paths

            for key, path in paths.items():
                assert Path(path).exists(), f"{key} file not created"

    def test_export_for_antspy(self):
        """Test comprehensive ANTsPy export with usage instructions."""
        shape = (8, 8, 8)
        displacement = np.random.randn(*shape, 3).astype(np.float32) * 0.5
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(
            displacement, affine, source_space="SUIT", target_space="patient"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ANTsTransformExporter(tmpdir)
            result = exporter.export_for_antspy(transform, base_filename="suit_to_patient")

            assert "paths" in result
            assert "source_space" in result
            assert "target_space" in result
            assert "usage_code" in result

            assert result["source_space"] == "SUIT"
            assert result["target_space"] == "patient"
            assert "ants.image_read" in result["usage_code"]

            # Check usage instructions file
            assert "usage_instructions" in result["paths"]
            usage_path = result["paths"]["usage_instructions"]
            assert Path(usage_path).exists()

    def test_metadata_json_export(self):
        """Test that metadata is properly exported as JSON."""
        shape = (8, 8, 8)
        displacement = np.random.randn(*shape, 3).astype(np.float32) * 0.5
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)
        transform.metadata = {
            "simulation_time": 30.0,
            "tumor_center": [10.0, 20.0, 30.0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ANTsTransformExporter(tmpdir)
            paths = exporter.export_composite_transform(transform, "test")

            meta_path = paths["metadata"]
            with open(meta_path) as f:
                metadata = json.load(f)

            assert "affine" in metadata
            assert "reference_shape" in metadata
            assert "max_displacement_mm" in metadata
            assert metadata["metadata"]["simulation_time"] == 30.0


class TestComputeTransformFromSimulation:
    """Tests for computing transforms from FEM simulation results."""

    def test_compute_transform_basic(self):
        """Test computing transform from node displacements."""
        # Create simple mesh nodes (grid of points)
        nodes = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    nodes.append([i * 2.0, j * 2.0, k * 2.0])
        nodes = np.array(nodes, dtype=np.float64)

        # Create displacements (uniform shift)
        displacements = np.ones_like(nodes) * 1.5

        volume_shape = (10, 10, 10)
        affine = np.eye(4)
        voxel_size = (1.0, 1.0, 1.0)

        transform = compute_transform_from_simulation(
            displacement_at_nodes=displacements,
            mesh_nodes=nodes,
            volume_shape=volume_shape,
            affine=affine,
            voxel_size=voxel_size,
            smoothing_sigma=1.0,
        )

        assert transform.reference_shape == volume_shape
        assert transform.displacement_field.shape == (*volume_shape, 3)
        assert transform.source_space == "SUIT"
        assert transform.target_space == "deformed"

        # Check that displacement is roughly 1.5 in regions near nodes
        max_disp = transform.get_max_displacement()
        assert max_disp > 0  # Should have some displacement

    def test_compute_transform_with_custom_affine(self):
        """Test computing transform with custom affine."""
        nodes = np.array([
            [5.0, 5.0, 5.0],
            [15.0, 15.0, 15.0],
        ], dtype=np.float64)
        displacements = np.array([
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ], dtype=np.float64)

        volume_shape = (20, 20, 20)
        affine = np.array([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1],
        ])
        voxel_size = (1.0, 1.0, 1.0)

        transform = compute_transform_from_simulation(
            displacement_at_nodes=displacements,
            mesh_nodes=nodes,
            volume_shape=volume_shape,
            affine=affine,
            voxel_size=voxel_size,
            smoothing_sigma=2.0,
        )

        assert np.allclose(transform.affine, affine)

    def test_compute_transform_zero_displacement(self):
        """Test with zero displacement returns valid transform."""
        nodes = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        displacements = np.zeros((1, 3), dtype=np.float64)

        volume_shape = (10, 10, 10)
        affine = np.eye(4)
        voxel_size = (1.0, 1.0, 1.0)

        transform = compute_transform_from_simulation(
            displacement_at_nodes=displacements,
            mesh_nodes=nodes,
            volume_shape=volume_shape,
            affine=affine,
            voxel_size=voxel_size,
        )

        assert transform.get_max_displacement() == 0.0


class TestTransformComposition:
    """Tests for transform composition."""

    def test_compose_identity_transforms(self):
        """Test composing two identity transforms."""
        shape = (8, 8, 8)
        voxel_size = (1.0, 1.0, 1.0)

        t1 = SpatialTransform.identity(shape, voxel_size)
        t2 = SpatialTransform.identity(shape, voxel_size)

        composed = t1.compose(t2)

        assert composed.reference_shape == shape
        assert np.allclose(composed.displacement_field, 0)

    def test_compose_with_uniform_displacement(self):
        """Test composing transforms with uniform displacements."""
        shape = (5, 5, 5)
        voxel_size = (1.0, 1.0, 1.0)
        affine = np.eye(4)

        # First transform: shift by (1, 0, 0)
        disp1 = np.ones((*shape, 3), dtype=np.float32)
        disp1[..., 0] = 1.0
        disp1[..., 1] = 0.0
        disp1[..., 2] = 0.0
        t1 = SpatialTransform.from_displacement_field(disp1, affine)

        # Second transform: shift by (0, 2, 0)
        disp2 = np.ones((*shape, 3), dtype=np.float32)
        disp2[..., 0] = 0.0
        disp2[..., 1] = 2.0
        disp2[..., 2] = 0.0
        t2 = SpatialTransform.from_displacement_field(disp2, affine)

        composed = t1.compose(t2)

        # The composed displacement should be approximately (1, 2, 0)
        # at points within the valid domain
        center_disp = composed.displacement_field[2, 2, 2]
        assert np.isclose(center_disp[0], 1.0, atol=0.1)
        assert np.isclose(center_disp[1], 2.0, atol=0.1)


class TestInverseTransform:
    """Tests for inverse transform computation."""

    def test_inverse_transform_small_displacement(self):
        """Test inverse transform with small displacement."""
        shape = (10, 10, 10)
        displacement = np.zeros((*shape, 3), dtype=np.float32)
        displacement[:, :, :] = [0.5, 0.3, 0.1]  # Small uniform displacement
        affine = np.eye(4)

        transform = SpatialTransform.from_displacement_field(displacement, affine)

        # Forward transform
        point = np.array([5.0, 5.0, 5.0])
        forward = transform.transform_point(point)

        # Inverse transform should approximately recover original
        recovered = transform.transform_point(forward, inverse=True)

        assert np.allclose(recovered, point, atol=0.1)
