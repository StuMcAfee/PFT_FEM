"""
Comprehensive tests for mesh generation module.

Tests cover:
- TetMesh data structure and operations
- MeshGenerator from various inputs
- Mesh quality metrics
- Mesh refinement
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pft_fem.mesh import TetMesh, MeshGenerator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def unit_tetrahedron():
    """Single unit tetrahedron mesh."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

    return TetMesh(nodes=nodes, elements=elements)


@pytest.fixture
def cube_mesh():
    """Unit cube decomposed into tetrahedra."""
    nodes = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=np.float64)

    # 5-tetrahedra decomposition
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
def sphere_mask():
    """Spherical binary mask for mesh generation tests."""
    shape = (15, 15, 15)
    center = np.array([7, 7, 7])
    radius = 5

    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    return dist <= radius


# =============================================================================
# TetMesh Property Tests
# =============================================================================


class TestTetMeshProperties:
    """Tests for TetMesh basic properties."""

    def test_num_nodes(self, unit_tetrahedron):
        """Test node count property."""
        assert unit_tetrahedron.num_nodes == 4

    def test_num_elements(self, unit_tetrahedron):
        """Test element count property."""
        assert unit_tetrahedron.num_elements == 1

    def test_num_nodes_cube(self, cube_mesh):
        """Test node count for cube mesh."""
        assert cube_mesh.num_nodes == 8

    def test_num_elements_cube(self, cube_mesh):
        """Test element count for cube mesh."""
        assert cube_mesh.num_elements == 5

    def test_node_shape(self, unit_tetrahedron):
        """Test that nodes have correct shape."""
        assert unit_tetrahedron.nodes.shape == (4, 3)

    def test_element_shape(self, unit_tetrahedron):
        """Test that elements have correct shape."""
        assert unit_tetrahedron.elements.shape == (1, 4)

    def test_element_indices_valid(self, cube_mesh):
        """Test that element indices are within valid range."""
        assert np.all(cube_mesh.elements >= 0)
        assert np.all(cube_mesh.elements < cube_mesh.num_nodes)


# =============================================================================
# TetMesh Volume Calculation Tests
# =============================================================================


class TestTetMeshVolumes:
    """Tests for element volume calculations."""

    def test_unit_tetrahedron_volume(self, unit_tetrahedron):
        """Test volume of unit tetrahedron."""
        volumes = unit_tetrahedron.compute_element_volumes()

        # Volume of tetrahedron with vertices at origin and unit vectors
        # V = |det([v1-v0, v2-v0, v3-v0])| / 6 = 1/6
        expected_volume = 1.0 / 6.0

        assert len(volumes) == 1
        assert_allclose(volumes[0], expected_volume, rtol=1e-10)

    def test_cube_total_volume(self, cube_mesh):
        """Test that cube mesh volumes sum to 1."""
        volumes = cube_mesh.compute_element_volumes()

        assert len(volumes) == 5
        assert np.all(volumes > 0)
        assert_allclose(np.sum(volumes), 1.0, rtol=1e-10)

    def test_scaled_tetrahedron_volume(self):
        """Test volume scales correctly with size."""
        scale = 2.0

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ], dtype=np.float64)

        mesh = TetMesh(
            nodes=nodes,
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        volumes = mesh.compute_element_volumes()

        # Volume scales as length^3
        expected = (scale**3) / 6.0
        assert_allclose(volumes[0], expected, rtol=1e-10)

    def test_all_volumes_positive(self, cube_mesh):
        """Test that all element volumes are positive."""
        volumes = cube_mesh.compute_element_volumes()
        assert np.all(volumes > 0)


# =============================================================================
# TetMesh Centroid Tests
# =============================================================================


class TestTetMeshCentroids:
    """Tests for element centroid calculations."""

    def test_unit_tetrahedron_centroid(self, unit_tetrahedron):
        """Test centroid of unit tetrahedron."""
        centroids = unit_tetrahedron.compute_element_centroids()

        # Centroid is average of vertices
        expected = np.array([[0.25, 0.25, 0.25]])

        assert centroids.shape == (1, 3)
        assert_allclose(centroids, expected, rtol=1e-10)

    def test_centroid_shape(self, cube_mesh):
        """Test centroid array shape."""
        centroids = cube_mesh.compute_element_centroids()
        assert centroids.shape == (5, 3)

    def test_centroids_inside_bounding_box(self, cube_mesh):
        """Test that centroids are within mesh bounding box."""
        centroids = cube_mesh.compute_element_centroids()

        min_coords = cube_mesh.nodes.min(axis=0)
        max_coords = cube_mesh.nodes.max(axis=0)

        assert np.all(centroids >= min_coords)
        assert np.all(centroids <= max_coords)


# =============================================================================
# TetMesh Node Operations Tests
# =============================================================================


class TestTetMeshNodeOperations:
    """Tests for node-related operations."""

    def test_get_element_nodes(self, unit_tetrahedron):
        """Test retrieving nodes for an element."""
        elem_nodes = unit_tetrahedron.get_element_nodes(0)

        assert elem_nodes.shape == (4, 3)
        assert_array_equal(elem_nodes, unit_tetrahedron.nodes)

    def test_find_nearest_node(self, cube_mesh):
        """Test finding nearest node to a point."""
        # Point near node 0 (origin)
        point = np.array([0.1, 0.1, 0.1])
        nearest = cube_mesh.find_nearest_node(point)
        assert nearest == 0

        # Point near node 7 (1,1,1)
        point = np.array([0.9, 0.9, 0.9])
        nearest = cube_mesh.find_nearest_node(point)
        assert nearest == 7

    def test_find_nearest_node_exact(self, cube_mesh):
        """Test finding node when query is exactly at a node."""
        for i in range(cube_mesh.num_nodes):
            nearest = cube_mesh.find_nearest_node(cube_mesh.nodes[i])
            assert nearest == i

    def test_find_nodes_in_sphere(self, cube_mesh):
        """Test finding nodes within a sphere."""
        center = np.array([0.5, 0.5, 0.5])
        radius = 0.6

        nodes_in_sphere = cube_mesh.find_nodes_in_sphere(center, radius)

        # No corner nodes are within 0.6 of cube center
        # (distance from center to corner is sqrt(0.5^2 * 3) ≈ 0.87)
        assert len(nodes_in_sphere) == 0

        # With larger radius, should find all nodes
        all_nodes = cube_mesh.find_nodes_in_sphere(center, 1.0)
        assert len(all_nodes) == 8

    def test_find_nodes_in_sphere_empty(self, cube_mesh):
        """Test finding nodes when no nodes in sphere."""
        # Sphere far from mesh
        center = np.array([10.0, 10.0, 10.0])
        radius = 1.0

        nodes = cube_mesh.find_nodes_in_sphere(center, radius)
        assert len(nodes) == 0


# =============================================================================
# TetMesh Neighbor Building Tests
# =============================================================================


class TestTetMeshNeighbors:
    """Tests for node neighbor computations."""

    def test_build_node_neighbors(self, unit_tetrahedron):
        """Test building node neighbor adjacency."""
        unit_tetrahedron.build_node_neighbors()

        # In a tetrahedron, every node is connected to every other node
        for i in range(4):
            neighbors = unit_tetrahedron.node_neighbors[i]
            assert len(neighbors) == 3
            assert i not in neighbors

    def test_neighbor_symmetry(self, cube_mesh):
        """Test that neighbor relationship is symmetric."""
        cube_mesh.build_node_neighbors()

        for i, neighbors in cube_mesh.node_neighbors.items():
            for j in neighbors:
                assert i in cube_mesh.node_neighbors[j]


# =============================================================================
# TetMesh Quality Metrics Tests
# =============================================================================


class TestTetMeshQuality:
    """Tests for mesh quality metrics."""

    def test_quality_metrics_keys(self, cube_mesh):
        """Test that quality metrics returns expected keys."""
        metrics = cube_mesh.compute_quality_metrics()

        expected_keys = [
            "num_nodes", "num_elements", "total_volume",
            "min_volume", "max_volume", "mean_volume",
            "min_aspect_ratio", "max_aspect_ratio", "mean_aspect_ratio",
        ]

        for key in expected_keys:
            assert key in metrics

    def test_quality_metrics_values(self, cube_mesh):
        """Test that quality metrics values are reasonable."""
        metrics = cube_mesh.compute_quality_metrics()

        assert metrics["num_nodes"] == 8
        assert metrics["num_elements"] == 5
        assert_allclose(metrics["total_volume"], 1.0, rtol=1e-10)

        assert metrics["min_volume"] > 0
        assert metrics["max_volume"] >= metrics["min_volume"]
        assert metrics["min_volume"] <= metrics["mean_volume"] <= metrics["max_volume"]

        assert metrics["min_aspect_ratio"] >= 1.0  # Aspect ratio always >= 1
        assert metrics["mean_aspect_ratio"] >= metrics["min_aspect_ratio"]

    def test_regular_tetrahedron_aspect_ratio(self):
        """Test aspect ratio of regular tetrahedron."""
        # Regular tetrahedron (equilateral)
        a = 1.0
        nodes = np.array([
            [0, 0, 0],
            [a, 0, 0],
            [a/2, a*np.sqrt(3)/2, 0],
            [a/2, a*np.sqrt(3)/6, a*np.sqrt(2/3)],
        ], dtype=np.float64)

        mesh = TetMesh(
            nodes=nodes,
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        metrics = mesh.compute_quality_metrics()

        # Regular tetrahedron has aspect ratio of 1
        assert_allclose(metrics["mean_aspect_ratio"], 1.0, rtol=1e-10)


# =============================================================================
# MeshGenerator Basic Tests
# =============================================================================


class TestMeshGeneratorBasic:
    """Basic tests for MeshGenerator."""

    def test_generator_initialization(self):
        """Test MeshGenerator initialization."""
        gen = MeshGenerator()

        assert gen.subdivision_method == "five"
        assert gen.min_edge_length == 1.0

    def test_generator_custom_params(self):
        """Test MeshGenerator with custom parameters."""
        gen = MeshGenerator(subdivision_method="six", min_edge_length=0.5)

        assert gen.subdivision_method == "six"
        assert gen.min_edge_length == 0.5


# =============================================================================
# MeshGenerator from_mask Tests
# =============================================================================


class TestMeshGeneratorFromMask:
    """Tests for mesh generation from binary masks."""

    def test_from_mask_single_voxel(self):
        """Test mesh generation from single voxel."""
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True

        gen = MeshGenerator()
        mesh = gen.from_mask(mask, simplify=False)

        # Single voxel should produce 5 tetrahedra (five decomposition)
        assert mesh.num_elements == 5
        # Single voxel has 8 corner nodes
        assert mesh.num_nodes == 8

    def test_from_mask_cube(self):
        """Test mesh generation from cube-shaped mask."""
        mask = np.ones((2, 2, 2), dtype=bool)

        gen = MeshGenerator()
        mesh = gen.from_mask(mask, simplify=False)

        # 8 voxels * 5 tets = 40 tetrahedra
        assert mesh.num_elements == 8 * 5

    def test_from_mask_sphere(self, sphere_mask):
        """Test mesh generation from spherical mask."""
        gen = MeshGenerator()
        mesh = gen.from_mask(sphere_mask, simplify=False)

        # Should have reasonable number of elements
        assert mesh.num_elements > 0
        assert mesh.num_nodes > 0

        # All element indices should be valid
        assert np.all(mesh.elements >= 0)
        assert np.all(mesh.elements < mesh.num_nodes)

    def test_from_mask_empty(self):
        """Test mesh generation from empty mask."""
        mask = np.zeros((5, 5, 5), dtype=bool)

        gen = MeshGenerator()
        mesh = gen.from_mask(mask)

        assert mesh.num_elements == 0
        assert mesh.num_nodes == 0

    def test_from_mask_with_voxel_size(self):
        """Test mesh generation with custom voxel size."""
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True

        voxel_size = (2.0, 2.0, 2.0)
        gen = MeshGenerator()
        mesh = gen.from_mask(mask, voxel_size=voxel_size, simplify=False)

        # Nodes should be scaled by voxel size
        assert np.max(mesh.nodes) == 4.0  # 2 voxels * 2mm
        assert np.min(mesh.nodes) == 2.0  # 1 voxel * 2mm

    def test_from_mask_with_anisotropic_voxels(self):
        """Test mesh generation with anisotropic voxels."""
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True

        voxel_size = (1.0, 2.0, 3.0)
        gen = MeshGenerator()
        mesh = gen.from_mask(mask, voxel_size=voxel_size, simplify=False)

        # Check nodes span correct ranges
        x_range = mesh.nodes[:, 0].max() - mesh.nodes[:, 0].min()
        y_range = mesh.nodes[:, 1].max() - mesh.nodes[:, 1].min()
        z_range = mesh.nodes[:, 2].max() - mesh.nodes[:, 2].min()

        assert_allclose(x_range, 1.0, rtol=1e-10)
        assert_allclose(y_range, 2.0, rtol=1e-10)
        assert_allclose(z_range, 3.0, rtol=1e-10)

    def test_from_mask_with_labels(self, sphere_mask):
        """Test mesh generation with label assignment."""
        labels = np.zeros_like(sphere_mask, dtype=np.int32)
        labels[sphere_mask] = 5

        gen = MeshGenerator()
        mesh = gen.from_mask(
            sphere_mask,
            labels=labels,
            simplify=False,
        )

        # Node labels should be assigned
        assert len(mesh.node_labels) == mesh.num_nodes
        # Most labels should be 5 (inside sphere)
        assert np.sum(mesh.node_labels == 5) > mesh.num_nodes * 0.5

    def test_from_mask_boundary_nodes(self, sphere_mask):
        """Test that boundary nodes are identified."""
        gen = MeshGenerator()
        mesh = gen.from_mask(sphere_mask, simplify=False)

        # Should have boundary nodes
        assert len(mesh.boundary_nodes) > 0
        assert len(mesh.boundary_nodes) < mesh.num_nodes

        # Boundary node indices should be valid
        assert np.all(mesh.boundary_nodes >= 0)
        assert np.all(mesh.boundary_nodes < mesh.num_nodes)


# =============================================================================
# MeshGenerator Subdivision Tests
# =============================================================================


class TestMeshGeneratorSubdivision:
    """Tests for mesh subdivision methods."""

    def test_five_vs_six_subdivision(self):
        """Test that different subdivisions give different element counts."""
        mask = np.ones((2, 2, 2), dtype=bool)

        gen_five = MeshGenerator(subdivision_method="five")
        mesh_five = gen_five.from_mask(mask, simplify=False)

        gen_six = MeshGenerator(subdivision_method="six")
        mesh_six = gen_six.from_mask(mask, simplify=False)

        # 5 vs 6 tetrahedra per voxel
        assert mesh_five.num_elements == 8 * 5
        assert mesh_six.num_elements == 8 * 6

    def test_subdivision_volume_preservation(self):
        """Test that both subdivisions preserve total volume."""
        mask = np.ones((3, 3, 3), dtype=bool)
        expected_volume = 27.0  # 27 voxels * 1 mm^3

        for method in ["five", "six"]:
            gen = MeshGenerator(subdivision_method=method)
            mesh = gen.from_mask(mask, voxel_size=(1.0, 1.0, 1.0), simplify=False)

            volumes = mesh.compute_element_volumes()
            total = np.sum(volumes)

            assert_allclose(total, expected_volume, rtol=1e-10)


# =============================================================================
# MeshGenerator from_surface Tests
# =============================================================================


class TestMeshGeneratorFromSurface:
    """Tests for mesh generation from surface points."""

    def test_from_surface_cube_corners(self):
        """Test Delaunay mesh from cube corner points."""
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float64)

        gen = MeshGenerator()
        mesh = gen.from_surface(points)

        assert mesh.num_nodes == 8
        assert mesh.num_elements > 0

        # All nodes should be original points
        assert_array_equal(mesh.nodes, points)

    def test_from_surface_sphere_points(self):
        """Test Delaunay mesh from sphere surface points."""
        # Generate points on sphere surface
        n_points = 50
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        theta = np.arccos(np.random.uniform(-1, 1, n_points))

        radius = 5.0
        points = np.column_stack([
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ])

        gen = MeshGenerator()
        mesh = gen.from_surface(points)

        assert mesh.num_nodes == n_points
        assert mesh.num_elements > 0


# =============================================================================
# MeshGenerator Refinement Tests
# =============================================================================


class TestMeshGeneratorRefinement:
    """Tests for mesh refinement operations."""

    def test_refine_region_increases_elements(self, cube_mesh):
        """Test that refinement increases element count."""
        gen = MeshGenerator()

        center = np.array([0.5, 0.5, 0.5])
        radius = 0.4

        refined = gen.refine_region(cube_mesh, center, radius)

        # Refined mesh should have more elements
        assert refined.num_elements >= cube_mesh.num_elements

    def test_refine_region_no_region(self, cube_mesh):
        """Test refinement when no elements in region."""
        gen = MeshGenerator()

        # Center far from mesh
        center = np.array([10.0, 10.0, 10.0])
        radius = 1.0

        refined = gen.refine_region(cube_mesh, center, radius)

        # Should be unchanged
        assert refined.num_elements == cube_mesh.num_elements

    def test_refine_region_preserves_volume(self, cube_mesh):
        """Test that refinement preserves total volume."""
        gen = MeshGenerator()

        original_volume = np.sum(cube_mesh.compute_element_volumes())

        center = np.array([0.5, 0.5, 0.5])
        refined = gen.refine_region(cube_mesh, center, radius=0.4)

        refined_volume = np.sum(refined.compute_element_volumes())

        assert_allclose(refined_volume, original_volume, rtol=1e-6)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestMeshEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_node_operations(self):
        """Test operations on mesh with single node."""
        # This is technically invalid but shouldn't crash
        nodes = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        elements = np.zeros((0, 4), dtype=np.int32)

        mesh = TetMesh(nodes=nodes, elements=elements)

        assert mesh.num_nodes == 1
        assert mesh.num_elements == 0

        # Should return empty arrays
        volumes = mesh.compute_element_volumes()
        centroids = mesh.compute_element_centroids()

        assert len(volumes) == 0
        assert len(centroids) == 0

    def test_degenerate_tetrahedron(self):
        """Test handling of degenerate (flat) tetrahedron."""
        # Coplanar nodes (degenerate)
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],  # In same plane
        ], dtype=np.float64)

        mesh = TetMesh(
            nodes=nodes,
            elements=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )

        volumes = mesh.compute_element_volumes()

        # Volume should be zero or near-zero
        assert_allclose(volumes[0], 0.0, atol=1e-10)

    def test_large_mesh_performance(self):
        """Test that mesh operations scale reasonably."""
        # Create larger mask
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:15, 5:15, 5:15] = True  # 10x10x10 cube

        gen = MeshGenerator()
        mesh = gen.from_mask(mask, simplify=False)

        # Should complete without timeout
        volumes = mesh.compute_element_volumes()
        centroids = mesh.compute_element_centroids()
        metrics = mesh.compute_quality_metrics()

        assert len(volumes) == mesh.num_elements
        assert len(centroids) == mesh.num_elements
        assert metrics["total_volume"] > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestMeshIntegration:
    """Integration tests combining mesh operations."""

    def test_full_mesh_workflow(self, sphere_mask):
        """Test complete mesh generation workflow."""
        # Generate mesh
        gen = MeshGenerator()
        mesh = gen.from_mask(
            sphere_mask,
            voxel_size=(1.0, 1.0, 1.0),
            simplify=False,
        )

        # Verify basic properties
        assert mesh.num_nodes > 0
        assert mesh.num_elements > 0

        # Compute all metrics
        volumes = mesh.compute_element_volumes()
        centroids = mesh.compute_element_centroids()
        metrics = mesh.compute_quality_metrics()

        # Verify volumes
        assert np.all(volumes > 0)
        assert_allclose(metrics["total_volume"], np.sum(volumes), rtol=1e-10)

        # Verify centroids are inside mesh bounds
        min_bounds = mesh.nodes.min(axis=0)
        max_bounds = mesh.nodes.max(axis=0)

        assert np.all(centroids >= min_bounds)
        assert np.all(centroids <= max_bounds)

        # Build neighbors and verify
        mesh.build_node_neighbors()
        assert len(mesh.node_neighbors) == mesh.num_nodes

        # Find nearest nodes
        center = np.mean(mesh.nodes, axis=0)
        nearest = mesh.find_nearest_node(center)
        assert 0 <= nearest < mesh.num_nodes

    def test_mesh_suitable_for_fem(self, sphere_mask):
        """Test that generated mesh is suitable for FEM."""
        gen = MeshGenerator()
        mesh = gen.from_mask(sphere_mask, simplify=False)

        # All volumes should be positive (proper orientation)
        volumes = mesh.compute_element_volumes()
        assert np.all(volumes > 0), "Some elements have non-positive volume"

        # No extremely small or large elements (quality check)
        volume_ratio = volumes.max() / volumes.min()
        assert volume_ratio < 1000, "Element size varies too much"

        # Aspect ratios should be reasonable
        metrics = mesh.compute_quality_metrics()
        assert metrics["max_aspect_ratio"] < 50, "Some elements too stretched"

        # Element connectivity should be valid
        assert np.all(mesh.elements >= 0)
        assert np.all(mesh.elements < mesh.num_nodes)

        # No duplicate elements
        elem_set = set(tuple(sorted(e)) for e in mesh.elements)
        assert len(elem_set) == mesh.num_elements, "Duplicate elements found"
