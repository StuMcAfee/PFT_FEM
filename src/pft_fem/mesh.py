"""
Mesh generation module for FEM analysis.

Provides utilities to generate tetrahedral meshes from volumetric atlas data
for finite element tumor growth simulation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class TetMesh:
    """
    Tetrahedral mesh data structure for FEM.

    Attributes:
        nodes: Node coordinates, shape (N, 3).
        elements: Element connectivity, shape (M, 4) for tetrahedra.
        node_labels: Tissue label at each node.
        boundary_nodes: Indices of nodes on the boundary.
        node_neighbors: Adjacency list for each node.
    """

    nodes: NDArray[np.float64]
    elements: NDArray[np.int32]
    node_labels: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    boundary_nodes: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    node_neighbors: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the mesh."""
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        """Number of elements in the mesh."""
        return len(self.elements)

    def compute_element_volumes(self) -> NDArray[np.float64]:
        """
        Compute volume of each tetrahedral element.

        Returns:
            Array of element volumes.
        """
        volumes = np.zeros(self.num_elements)

        for i, elem in enumerate(self.elements):
            v0, v1, v2, v3 = self.nodes[elem]
            # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
            mat = np.array([v1 - v0, v2 - v0, v3 - v0])
            volumes[i] = np.abs(np.linalg.det(mat)) / 6.0

        return volumes

    def compute_element_centroids(self) -> NDArray[np.float64]:
        """
        Compute centroid of each element.

        Returns:
            Array of centroids, shape (M, 3).
        """
        centroids = np.zeros((self.num_elements, 3))

        for i, elem in enumerate(self.elements):
            centroids[i] = np.mean(self.nodes[elem], axis=0)

        return centroids

    def get_element_nodes(self, element_idx: int) -> NDArray[np.float64]:
        """Get node coordinates for a specific element."""
        return self.nodes[self.elements[element_idx]]

    def build_node_neighbors(self) -> None:
        """Build node neighbor adjacency information."""
        self.node_neighbors = {i: [] for i in range(self.num_nodes)}

        for elem in self.elements:
            for i in range(4):
                for j in range(i + 1, 4):
                    ni, nj = elem[i], elem[j]
                    if nj not in self.node_neighbors[ni]:
                        self.node_neighbors[ni].append(nj)
                    if ni not in self.node_neighbors[nj]:
                        self.node_neighbors[nj].append(ni)

    def compute_quality_metrics(self) -> Dict[str, float]:
        """
        Compute mesh quality metrics.

        Returns:
            Dictionary with quality statistics.
        """
        volumes = self.compute_element_volumes()

        # Aspect ratio for tetrahedra
        aspect_ratios = []
        for elem in self.elements:
            coords = self.nodes[elem]
            # Compute edge lengths
            edges = []
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append(np.linalg.norm(coords[i] - coords[j]))
            edges = np.array(edges)
            aspect_ratios.append(edges.max() / edges.min())

        aspect_ratios = np.array(aspect_ratios)

        return {
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "total_volume": float(np.sum(volumes)),
            "min_volume": float(np.min(volumes)),
            "max_volume": float(np.max(volumes)),
            "mean_volume": float(np.mean(volumes)),
            "min_aspect_ratio": float(np.min(aspect_ratios)),
            "max_aspect_ratio": float(np.max(aspect_ratios)),
            "mean_aspect_ratio": float(np.mean(aspect_ratios)),
            # Aliases for backward compatibility (quality = 1/aspect_ratio, normalized)
            "min_quality": float(1.0 / np.max(aspect_ratios)),
            "max_quality": float(1.0 / np.min(aspect_ratios)),
            "mean_quality": float(1.0 / np.mean(aspect_ratios)),
        }

    def find_nearest_node(self, point: NDArray[np.float64]) -> int:
        """Find the node nearest to a given point."""
        distances = np.linalg.norm(self.nodes - point, axis=1)
        return int(np.argmin(distances))

    def find_nodes_in_sphere(
        self,
        center: NDArray[np.float64],
        radius: float,
    ) -> NDArray[np.int32]:
        """Find all nodes within a sphere."""
        distances = np.linalg.norm(self.nodes - center, axis=1)
        return np.where(distances <= radius)[0].astype(np.int32)


class MeshGenerator:
    """
    Generator for creating tetrahedral meshes from volumetric data.

    Implements a simple voxel-to-tetrahedra conversion for FEM analysis.
    Each voxel is subdivided into 5 or 6 tetrahedra.
    """

    def __init__(
        self,
        subdivision_method: str = "five",
        min_edge_length: float = 1.0,
    ):
        """
        Initialize mesh generator.

        Args:
            subdivision_method: How to subdivide voxels ("five" or "six").
            min_edge_length: Minimum edge length for mesh refinement.
        """
        self.subdivision_method = subdivision_method
        self.min_edge_length = min_edge_length

    def from_mask(
        self,
        mask: NDArray[np.bool_],
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        labels: Optional[NDArray[np.int32]] = None,
        affine: Optional[NDArray[np.float64]] = None,
        simplify: bool = True,
        target_reduction: float = 0.5,
    ) -> TetMesh:
        """
        Generate tetrahedral mesh from a binary mask.

        Args:
            mask: Binary mask defining the region to mesh.
            voxel_size: Physical size of each voxel in mm.
            labels: Optional label volume for tissue assignment.
            affine: Optional affine transformation matrix.
            simplify: Whether to simplify the mesh.
            target_reduction: Target reduction ratio for simplification.

        Returns:
            TetMesh object.
        """
        # Find voxels to mesh
        voxel_coords = np.array(np.where(mask)).T

        if len(voxel_coords) == 0:
            return TetMesh(
                nodes=np.zeros((0, 3), dtype=np.float64),
                elements=np.zeros((0, 4), dtype=np.int32),
            )

        # Build nodes at voxel corners
        nodes, node_map = self._build_corner_nodes(voxel_coords, voxel_size)

        # Convert to physical coordinates
        if affine is not None:
            # Apply affine transformation
            nodes_homogeneous = np.hstack([nodes, np.ones((len(nodes), 1))])
            nodes = (affine @ nodes_homogeneous.T).T[:, :3]

        # Generate tetrahedra
        elements = self._voxels_to_tetrahedra(voxel_coords, node_map)

        # Assign labels to nodes
        if labels is not None:
            node_labels = self._assign_node_labels(
                nodes, voxel_coords, labels, voxel_size, affine
            )
        else:
            node_labels = np.ones(len(nodes), dtype=np.int32)

        # Find boundary nodes
        boundary_nodes = self._find_boundary_nodes(mask, voxel_coords, node_map)

        mesh = TetMesh(
            nodes=nodes,
            elements=elements,
            node_labels=node_labels,
            boundary_nodes=boundary_nodes,
        )

        # Optionally simplify mesh
        if simplify and len(elements) > 1000:
            mesh = self._simplify_mesh(mesh, target_reduction)

        return mesh

    def _build_corner_nodes(
        self,
        voxel_coords: NDArray[np.int32],
        voxel_size: Tuple[float, float, float],
    ) -> Tuple[NDArray[np.float64], Dict[Tuple[int, int, int], int]]:
        """Build unique nodes at voxel corners."""
        node_set = set()
        voxel_size = np.array(voxel_size)

        # Each voxel contributes 8 corners
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ])

        for voxel in voxel_coords:
            for corner in corners:
                node_set.add(tuple(voxel + corner))

        # Create node array and mapping
        node_list = sorted(node_set)
        node_map = {coord: idx for idx, coord in enumerate(node_list)}

        nodes = np.array(node_list, dtype=np.float64) * voxel_size

        return nodes, node_map

    def _voxels_to_tetrahedra(
        self,
        voxel_coords: NDArray[np.int32],
        node_map: Dict[Tuple[int, int, int], int],
    ) -> NDArray[np.int32]:
        """Convert voxels to tetrahedra."""
        elements = []

        # Corner offsets for a voxel (ordered for consistent subdivision)
        # 0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (1,1,0)
        # 4: (0,0,1), 5: (1,0,1), 6: (0,1,1), 7: (1,1,1)
        corners = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]

        # Five-tetrahedra decomposition (consistent orientation)
        if self.subdivision_method == "five":
            tet_patterns = [
                [0, 1, 3, 5],
                [0, 3, 2, 6],
                [0, 5, 4, 6],
                [3, 5, 6, 7],
                [0, 3, 5, 6],
            ]
        else:  # six tetrahedra
            tet_patterns = [
                [0, 1, 2, 4],
                [1, 2, 4, 5],
                [2, 4, 5, 6],
                [1, 2, 3, 5],
                [2, 3, 5, 6],
                [3, 5, 6, 7],
            ]

        for voxel in voxel_coords:
            # Get corner node indices for this voxel
            corner_nodes = [
                node_map[tuple(voxel + np.array(c))] for c in corners
            ]

            # Create tetrahedra
            for pattern in tet_patterns:
                tet = [corner_nodes[i] for i in pattern]
                elements.append(tet)

        return np.array(elements, dtype=np.int32)

    def _assign_node_labels(
        self,
        nodes: NDArray[np.float64],
        voxel_coords: NDArray[np.int32],
        labels: NDArray[np.int32],
        voxel_size: Tuple[float, float, float],
        affine: Optional[NDArray[np.float64]],
    ) -> NDArray[np.int32]:
        """Assign tissue labels to nodes."""
        node_labels = np.zeros(len(nodes), dtype=np.int32)

        # Convert node coordinates back to voxel space
        if affine is not None:
            inv_affine = np.linalg.inv(affine)
            nodes_homogeneous = np.hstack([nodes, np.ones((len(nodes), 1))])
            voxel_space = (inv_affine @ nodes_homogeneous.T).T[:, :3]
        else:
            voxel_space = nodes / np.array(voxel_size)

        # Round to nearest voxel
        voxel_indices = np.round(voxel_space).astype(np.int32)

        # Clamp to valid range
        for dim in range(3):
            voxel_indices[:, dim] = np.clip(
                voxel_indices[:, dim], 0, labels.shape[dim] - 1
            )

        # Look up labels
        for i, voxel in enumerate(voxel_indices):
            node_labels[i] = labels[voxel[0], voxel[1], voxel[2]]

        return node_labels

    def _find_boundary_nodes(
        self,
        mask: NDArray[np.bool_],
        voxel_coords: NDArray[np.int32],
        node_map: Dict[Tuple[int, int, int], int],
    ) -> NDArray[np.int32]:
        """Find nodes on the mesh boundary."""
        from scipy import ndimage

        # Find boundary voxels
        eroded = ndimage.binary_erosion(mask)
        boundary_mask = mask & ~eroded

        boundary_nodes = set()

        corners = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]

        for voxel in voxel_coords:
            if boundary_mask[voxel[0], voxel[1], voxel[2]]:
                for corner in corners:
                    coord = tuple(voxel + np.array(corner))
                    if coord in node_map:
                        boundary_nodes.add(node_map[coord])

        return np.array(sorted(boundary_nodes), dtype=np.int32)

    def _simplify_mesh(
        self,
        mesh: TetMesh,
        target_reduction: float,
    ) -> TetMesh:
        """
        Simplify mesh by removing small elements and merging close nodes.

        This is a simple implementation - for production use, consider
        using a dedicated mesh library like meshio or pygmsh.
        """
        # For now, just return the mesh as-is
        # A full implementation would use quadric error metrics or similar
        return mesh

    def from_surface(
        self,
        surface_points: NDArray[np.float64],
        resolution: float = 2.0,
    ) -> TetMesh:
        """
        Generate tetrahedral mesh from surface points using Delaunay.

        Args:
            surface_points: Surface point cloud, shape (N, 3).
            resolution: Target element size.

        Returns:
            TetMesh object.
        """
        from scipy.spatial import Delaunay

        # Use Delaunay triangulation
        tri = Delaunay(surface_points)

        return TetMesh(
            nodes=surface_points,
            elements=tri.simplices.astype(np.int32),
        )

    def refine_region(
        self,
        mesh: TetMesh,
        center: NDArray[np.float64],
        radius: float,
        refinement_factor: int = 2,
    ) -> TetMesh:
        """
        Refine mesh in a spherical region (e.g., around a tumor).

        Args:
            mesh: Input mesh.
            center: Center of refinement region.
            radius: Radius of refinement region.
            refinement_factor: How many times to subdivide elements.

        Returns:
            Refined mesh.
        """
        # Find elements in the region
        centroids = mesh.compute_element_centroids()
        distances = np.linalg.norm(centroids - center, axis=1)
        elements_to_refine = np.where(distances <= radius)[0]

        if len(elements_to_refine) == 0:
            return mesh

        # Subdivide selected elements
        new_nodes = list(mesh.nodes)
        new_elements = []
        node_map = {}  # Maps edge midpoint to new node index

        for i, elem in enumerate(mesh.elements):
            if i in elements_to_refine:
                # Subdivide this element
                subdivided = self._subdivide_tetrahedron(
                    elem, mesh.nodes, new_nodes, node_map
                )
                new_elements.extend(subdivided)
            else:
                new_elements.append(elem.tolist())

        return TetMesh(
            nodes=np.array(new_nodes, dtype=np.float64),
            elements=np.array(new_elements, dtype=np.int32),
            node_labels=mesh.node_labels,  # Labels need recomputing
            boundary_nodes=mesh.boundary_nodes,
        )

    def _subdivide_tetrahedron(
        self,
        elem: NDArray[np.int32],
        nodes: NDArray[np.float64],
        new_nodes: List[NDArray[np.float64]],
        node_map: Dict[Tuple[int, int], int],
    ) -> List[List[int]]:
        """Subdivide a tetrahedron into 8 smaller tetrahedra."""
        # Get edge midpoints
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        midpoints = []

        for e in edges:
            edge_key = tuple(sorted([elem[e[0]], elem[e[1]]]))
            if edge_key not in node_map:
                midpoint = (nodes[elem[e[0]]] + nodes[elem[e[1]]]) / 2
                new_nodes.append(midpoint)
                node_map[edge_key] = len(new_nodes) - 1
            midpoints.append(node_map[edge_key])

        # Create 8 new tetrahedra
        # Node indices: 0,1,2,3 = original corners
        # m01, m02, m03, m12, m13, m23 = midpoints
        n = elem.tolist()
        m = midpoints  # m01, m02, m03, m12, m13, m23

        new_tets = [
            [n[0], m[0], m[1], m[2]],  # Corner 0
            [n[1], m[0], m[3], m[4]],  # Corner 1
            [n[2], m[1], m[3], m[5]],  # Corner 2
            [n[3], m[2], m[4], m[5]],  # Corner 3
            # Interior tetrahedra (octahedron subdivision)
            [m[0], m[1], m[2], m[3]],
            [m[0], m[2], m[3], m[4]],
            [m[2], m[3], m[4], m[5]],
            [m[1], m[2], m[3], m[5]],
        ]

        return new_tets


def save_mesh(mesh: TetMesh, filename: str) -> None:
    """
    Save mesh to file using meshio.

    Args:
        mesh: Mesh to save.
        filename: Output filename (supports .vtk, .vtu, .msh, etc.).
    """
    import meshio

    cells = [("tetra", mesh.elements)]

    # Add point data
    point_data = {}
    if len(mesh.node_labels) > 0:
        point_data["labels"] = mesh.node_labels

    meshio_mesh = meshio.Mesh(
        points=mesh.nodes,
        cells=cells,
        point_data=point_data if point_data else None,
    )

    meshio_mesh.write(filename)


def load_mesh(filename: str) -> TetMesh:
    """
    Load mesh from file using meshio.

    Args:
        filename: Input filename.

    Returns:
        Loaded TetMesh.
    """
    import meshio

    meshio_mesh = meshio.read(filename)

    # Extract tetrahedra
    elements = None
    for cell_block in meshio_mesh.cells:
        if cell_block.type == "tetra":
            elements = cell_block.data.astype(np.int32)
            break

    if elements is None:
        raise ValueError("No tetrahedral elements found in mesh file")

    # Extract labels if present
    node_labels = np.array([], dtype=np.int32)
    if meshio_mesh.point_data and "labels" in meshio_mesh.point_data:
        node_labels = meshio_mesh.point_data["labels"].astype(np.int32)

    return TetMesh(
        nodes=meshio_mesh.points.astype(np.float64),
        elements=elements,
        node_labels=node_labels,
    )
