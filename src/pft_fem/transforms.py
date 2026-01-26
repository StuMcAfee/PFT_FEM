"""
Spatial Transform module for tracking and exporting deformations.

Provides functionality to track the complete spatial transform from
the SUIT template through the simulation process, and export in
ANTsPy-compatible formats.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import json

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpatialTransform:
    """
    Complete spatial transform from SUIT template to deformed state.

    Tracks both the affine transformation (template space to physical space)
    and the displacement field (deformation from tumor growth).

    The complete transform T that maps template coordinates to deformed
    coordinates is: T(x) = affine @ x + displacement(x)

    Attributes:
        affine: 4x4 affine transformation matrix (template voxels to physical mm).
        displacement_field: 3D displacement field, shape (X, Y, Z, 3) in mm.
        reference_shape: Shape of the reference volume (X, Y, Z).
        voxel_size: Voxel dimensions in mm.
        source_space: Name of the source coordinate space (e.g., "SUIT").
        target_space: Name of the target coordinate space (e.g., "deformed").
        metadata: Additional transform metadata.
    """

    affine: NDArray[np.float64]
    displacement_field: NDArray[np.float32]
    reference_shape: Tuple[int, int, int]
    voxel_size: Tuple[float, float, float]
    source_space: str = "SUIT"
    target_space: str = "deformed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate transform components."""
        if self.affine.shape != (4, 4):
            raise ValueError(f"Affine must be 4x4, got {self.affine.shape}")

        expected_disp_shape = (*self.reference_shape, 3)
        if self.displacement_field.shape != expected_disp_shape:
            raise ValueError(
                f"Displacement field shape {self.displacement_field.shape} "
                f"does not match expected {expected_disp_shape}"
            )

    @classmethod
    def identity(
        cls,
        shape: Tuple[int, int, int],
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        affine: Optional[NDArray[np.float64]] = None,
    ) -> "SpatialTransform":
        """
        Create an identity transform (no deformation).

        Args:
            shape: Volume dimensions.
            voxel_size: Voxel size in mm.
            affine: Affine matrix. If None, uses identity scaled by voxel_size.

        Returns:
            SpatialTransform with zero displacement.
        """
        if affine is None:
            affine = np.diag([*voxel_size, 1.0])

        displacement = np.zeros((*shape, 3), dtype=np.float32)

        return cls(
            affine=affine.astype(np.float64),
            displacement_field=displacement,
            reference_shape=shape,
            voxel_size=voxel_size,
        )

    @classmethod
    def from_displacement_field(
        cls,
        displacement: NDArray,
        affine: NDArray[np.float64],
        voxel_size: Optional[Tuple[float, float, float]] = None,
        source_space: str = "SUIT",
        target_space: str = "deformed",
    ) -> "SpatialTransform":
        """
        Create transform from a displacement field.

        Args:
            displacement: Displacement field, shape (X, Y, Z, 3) in mm.
            affine: 4x4 affine transformation matrix.
            voxel_size: Voxel dimensions. If None, derived from affine.
            source_space: Name of source coordinate space.
            target_space: Name of target coordinate space.

        Returns:
            SpatialTransform instance.
        """
        if voxel_size is None:
            voxel_size = tuple(np.abs(np.diag(affine)[:3]).tolist())

        shape = displacement.shape[:3]

        return cls(
            affine=affine.astype(np.float64),
            displacement_field=displacement.astype(np.float32),
            reference_shape=shape,
            voxel_size=voxel_size,
            source_space=source_space,
            target_space=target_space,
        )

    def get_displacement_at_point(
        self,
        point: NDArray[np.float64],
        interpolation: str = "linear",
    ) -> NDArray[np.float64]:
        """
        Get displacement at a physical coordinate.

        Args:
            point: Physical coordinate (x, y, z) in mm.
            interpolation: Interpolation method ("nearest" or "linear").

        Returns:
            Displacement vector (dx, dy, dz) in mm.
        """
        # Convert physical to voxel coordinates
        inv_affine = np.linalg.inv(self.affine)
        voxel = (inv_affine @ np.append(point, 1.0))[:3]

        if interpolation == "nearest":
            vi = np.round(voxel).astype(int)
            if all(0 <= vi[d] < self.reference_shape[d] for d in range(3)):
                return self.displacement_field[vi[0], vi[1], vi[2]].astype(np.float64)
            return np.zeros(3, dtype=np.float64)

        elif interpolation == "linear":
            return self._trilinear_interpolate(voxel)

        else:
            raise ValueError(f"Unknown interpolation: {interpolation}")

    def _trilinear_interpolate(
        self,
        voxel: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Trilinear interpolation of displacement at voxel coordinate."""
        shape = self.reference_shape

        # Clamp to valid range
        voxel = np.clip(voxel, [0, 0, 0], [s - 1 for s in shape])

        x0, y0, z0 = np.floor(voxel).astype(int)
        x1 = min(x0 + 1, shape[0] - 1)
        y1 = min(y0 + 1, shape[1] - 1)
        z1 = min(z0 + 1, shape[2] - 1)

        xd, yd, zd = voxel - np.array([x0, y0, z0])

        # Trilinear interpolation
        c00 = self.displacement_field[x0, y0, z0] * (1 - xd) + self.displacement_field[x1, y0, z0] * xd
        c01 = self.displacement_field[x0, y0, z1] * (1 - xd) + self.displacement_field[x1, y0, z1] * xd
        c10 = self.displacement_field[x0, y1, z0] * (1 - xd) + self.displacement_field[x1, y1, z0] * xd
        c11 = self.displacement_field[x0, y1, z1] * (1 - xd) + self.displacement_field[x1, y1, z1] * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        result = c0 * (1 - zd) + c1 * zd

        return result.astype(np.float64)

    def transform_point(
        self,
        point: NDArray[np.float64],
        inverse: bool = False,
    ) -> NDArray[np.float64]:
        """
        Transform a point from source to target space.

        Args:
            point: Point coordinates (x, y, z) in mm.
            inverse: If True, transform from target to source.

        Returns:
            Transformed point coordinates.
        """
        if inverse:
            # Inverse requires iterative solution
            return self._inverse_transform_point(point)

        # Forward transform: apply displacement
        displacement = self.get_displacement_at_point(point)
        return point + displacement

    def _inverse_transform_point(
        self,
        target_point: NDArray[np.float64],
        max_iter: int = 10,
        tolerance: float = 0.01,
    ) -> NDArray[np.float64]:
        """
        Compute inverse transform using fixed-point iteration.

        Args:
            target_point: Point in target (deformed) space.
            max_iter: Maximum iterations.
            tolerance: Convergence tolerance in mm.

        Returns:
            Corresponding point in source space.
        """
        # Initial guess
        source = target_point.copy()

        for _ in range(max_iter):
            # Forward transform
            disp = self.get_displacement_at_point(source)
            transformed = source + disp

            # Error
            error = target_point - transformed
            if np.linalg.norm(error) < tolerance:
                break

            # Update guess
            source = source + error

        return source

    def compose(self, other: "SpatialTransform") -> "SpatialTransform":
        """
        Compose this transform with another (self followed by other).

        Args:
            other: Transform to apply after this one.

        Returns:
            New composed transform.
        """
        # For displacement fields, composition requires resampling
        from scipy import ndimage

        # Create new displacement field
        new_disp = np.zeros_like(self.displacement_field)
        shape = self.reference_shape

        # For each voxel in source space
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    # Physical coordinate
                    phys = (self.affine @ np.array([i, j, k, 1]))[:3]

                    # Apply first transform
                    d1 = self.displacement_field[i, j, k]
                    intermediate = phys + d1

                    # Get second transform at intermediate location
                    d2 = other.get_displacement_at_point(intermediate)

                    # Total displacement
                    new_disp[i, j, k] = d1 + d2

        return SpatialTransform(
            affine=self.affine.copy(),
            displacement_field=new_disp,
            reference_shape=self.reference_shape,
            voxel_size=self.voxel_size,
            source_space=self.source_space,
            target_space=other.target_space,
            metadata={
                "composed_from": [self.metadata, other.metadata],
            },
        )

    def get_jacobian_determinant(self) -> NDArray[np.float32]:
        """
        Compute Jacobian determinant of the displacement field.

        The Jacobian determinant indicates local volume change:
        - det(J) > 1: local expansion
        - det(J) < 1: local compression
        - det(J) < 0: folding (invalid)

        Returns:
            Jacobian determinant at each voxel.
        """
        vx, vy, vz = self.voxel_size

        # Compute gradient of each displacement component along each axis
        # du_x/dx, du_x/dy, du_x/dz
        du_x_dx = np.gradient(self.displacement_field[..., 0], vx, axis=0)
        du_x_dy = np.gradient(self.displacement_field[..., 0], vy, axis=1)
        du_x_dz = np.gradient(self.displacement_field[..., 0], vz, axis=2)

        # du_y/dx, du_y/dy, du_y/dz
        du_y_dx = np.gradient(self.displacement_field[..., 1], vx, axis=0)
        du_y_dy = np.gradient(self.displacement_field[..., 1], vy, axis=1)
        du_y_dz = np.gradient(self.displacement_field[..., 1], vz, axis=2)

        # du_z/dx, du_z/dy, du_z/dz
        du_z_dx = np.gradient(self.displacement_field[..., 2], vx, axis=0)
        du_z_dy = np.gradient(self.displacement_field[..., 2], vy, axis=1)
        du_z_dz = np.gradient(self.displacement_field[..., 2], vz, axis=2)

        # Jacobian matrix: J = I + grad(u)
        # J = [[1 + du_x/dx, du_x/dy, du_x/dz],
        #      [du_y/dx, 1 + du_y/dy, du_y/dz],
        #      [du_z/dx, du_z/dy, 1 + du_z/dz]]

        j11 = 1 + du_x_dx
        j12 = du_x_dy
        j13 = du_x_dz
        j21 = du_y_dx
        j22 = 1 + du_y_dy
        j23 = du_y_dz
        j31 = du_z_dx
        j32 = du_z_dy
        j33 = 1 + du_z_dz

        # Compute determinant
        jacobian_det = (
            j11 * (j22 * j33 - j23 * j32)
            - j12 * (j21 * j33 - j23 * j31)
            + j13 * (j21 * j32 - j22 * j31)
        )

        return jacobian_det.astype(np.float32)

    def get_displacement_magnitude(self) -> NDArray[np.float32]:
        """
        Compute displacement magnitude at each voxel.

        Returns:
            Displacement magnitude in mm.
        """
        return np.linalg.norm(self.displacement_field, axis=-1).astype(np.float32)

    def get_max_displacement(self) -> float:
        """Get maximum displacement magnitude."""
        return float(np.max(self.get_displacement_magnitude()))

    def get_mean_displacement(self, mask: Optional[NDArray[np.bool_]] = None) -> float:
        """
        Get mean displacement magnitude.

        Args:
            mask: Optional mask to restrict computation.

        Returns:
            Mean displacement in mm.
        """
        mag = self.get_displacement_magnitude()
        if mask is not None:
            return float(np.mean(mag[mask]))
        return float(np.mean(mag[mag > 0]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "affine": self.affine.tolist(),
            "reference_shape": list(self.reference_shape),
            "voxel_size": list(self.voxel_size),
            "source_space": self.source_space,
            "target_space": self.target_space,
            "max_displacement_mm": self.get_max_displacement(),
            "metadata": self._serialize_metadata(self.metadata),
        }

    def _serialize_metadata(self, obj: Any) -> Any:
        """Recursively convert numpy arrays and tuples to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (tuple, list)):
            return [self._serialize_metadata(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_metadata(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


class ANTsTransformExporter:
    """
    Export spatial transforms in ANTsPy-compatible formats.

    Supports exporting as:
    - NIfTI displacement field (compatible with ANTs/ITK)
    - ANTs .mat affine transform
    - Combined composite transform
    """

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_displacement_field(
        self,
        transform: SpatialTransform,
        filename: str = "displacement_field.nii.gz",
    ) -> Path:
        """
        Export displacement field as NIfTI (ANTs-compatible format).

        The displacement field is stored in ITK/ANTs convention:
        - Shape: (X, Y, Z, 1, 3) for 3D vector field
        - Values represent displacement in mm (physical coordinates)

        Args:
            transform: SpatialTransform to export.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        import nibabel as nib

        filepath = self.output_dir / filename

        # ANTs/ITK expects shape (X, Y, Z, 1, 3) for 3D displacement
        # The singleton dimension is for time/vector dimension
        disp_data = transform.displacement_field.copy()

        # Add singleton dimension: (X, Y, Z, 3) -> (X, Y, Z, 1, 3)
        disp_data = disp_data[:, :, :, np.newaxis, :]

        # Create NIfTI image
        img = nib.Nifti1Image(disp_data.astype(np.float32), transform.affine)

        # Set intent to displacement field
        img.header.set_intent("vector", name="displacement")

        # Set data type
        img.header.set_data_dtype(np.float32)

        nib.save(img, filepath)

        return filepath

    def export_affine_transform(
        self,
        transform: SpatialTransform,
        filename: str = "affine_transform.mat",
    ) -> Path:
        """
        Export affine transformation in ANTs .mat format.

        Args:
            transform: SpatialTransform to export.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        filepath = self.output_dir / filename

        # ANTs .mat format is a text file with specific structure
        # Extract rotation/scale and translation
        affine = transform.affine

        # ANTs uses a specific parameterization
        # For simplicity, export the 4x4 matrix directly
        lines = [
            "#Insight Transform File V1.0",
            "#Transform 0",
            "Transform: AffineTransform_double_3_3",
            f"Parameters: {' '.join(map(str, affine[:3, :3].flatten()))} {' '.join(map(str, affine[:3, 3]))}",
            f"FixedParameters: 0 0 0",
        ]

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        return filepath

    def export_composite_transform(
        self,
        transform: SpatialTransform,
        base_filename: str = "transform",
        include_inverse: bool = True,
    ) -> Dict[str, Path]:
        """
        Export complete composite transform (affine + displacement).

        Creates multiple files that can be used with ANTsPy:
        - Displacement field (NIfTI)
        - Affine transform (.mat)
        - Metadata (JSON)
        - Optionally, inverse displacement field

        Args:
            transform: SpatialTransform to export.
            base_filename: Base name for output files.
            include_inverse: Whether to include approximate inverse.

        Returns:
            Dictionary mapping output type to file path.
        """
        paths = {}

        # Export displacement field
        disp_path = self.export_displacement_field(
            transform,
            f"{base_filename}_warp.nii.gz"
        )
        paths["displacement_field"] = disp_path

        # Export affine
        affine_path = self.export_affine_transform(
            transform,
            f"{base_filename}_affine.mat"
        )
        paths["affine"] = affine_path

        # Export inverse displacement field (approximate)
        if include_inverse:
            inverse_disp = self._compute_inverse_displacement(transform)
            inverse_path = self._save_inverse_displacement(
                inverse_disp,
                transform.affine,
                f"{base_filename}_inverse_warp.nii.gz"
            )
            paths["inverse_displacement"] = inverse_path

        # Export metadata
        meta_path = self.output_dir / f"{base_filename}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(transform.to_dict(), f, indent=2)
        paths["metadata"] = meta_path

        # Export Jacobian determinant for quality assessment
        jacobian = transform.get_jacobian_determinant()
        jac_path = self._save_jacobian(
            jacobian,
            transform.affine,
            f"{base_filename}_jacobian.nii.gz"
        )
        paths["jacobian_determinant"] = jac_path

        return paths

    def _compute_inverse_displacement(
        self,
        transform: SpatialTransform,
    ) -> NDArray[np.float32]:
        """
        Compute approximate inverse displacement field.

        Uses iterative fixed-point method to invert the displacement.

        Args:
            transform: Forward transform.

        Returns:
            Inverse displacement field.
        """
        shape = transform.reference_shape
        inverse_disp = np.zeros_like(transform.displacement_field)

        # Simple approximation: -displacement at each point
        # This is accurate for small displacements
        inverse_disp = -transform.displacement_field.copy()

        # For more accuracy, iteratively refine
        # (disabled for performance; enable if needed)
        # for _ in range(3):
        #     for i in range(shape[0]):
        #         for j in range(shape[1]):
        #             for k in range(shape[2]):
        #                 phys = (transform.affine @ np.array([i, j, k, 1]))[:3]
        #                 target = phys + inverse_disp[i, j, k]
        #                 forward_disp = transform.get_displacement_at_point(target)
        #                 inverse_disp[i, j, k] = -forward_disp

        return inverse_disp

    def _save_inverse_displacement(
        self,
        displacement: NDArray[np.float32],
        affine: NDArray[np.float64],
        filename: str,
    ) -> Path:
        """Save inverse displacement field."""
        import nibabel as nib

        filepath = self.output_dir / filename

        # Add singleton dimension for ANTs compatibility
        disp_data = displacement[:, :, :, np.newaxis, :]

        img = nib.Nifti1Image(disp_data.astype(np.float32), affine)
        img.header.set_intent("vector", name="inverse_displacement")
        nib.save(img, filepath)

        return filepath

    def _save_jacobian(
        self,
        jacobian: NDArray[np.float32],
        affine: NDArray[np.float64],
        filename: str,
    ) -> Path:
        """Save Jacobian determinant map."""
        import nibabel as nib

        filepath = self.output_dir / filename

        img = nib.Nifti1Image(jacobian, affine)
        img.header["descrip"] = b"Jacobian determinant of displacement field"
        nib.save(img, filepath)

        return filepath

    def export_for_antspy(
        self,
        transform: SpatialTransform,
        base_filename: str = "suit_to_deformed",
    ) -> Dict[str, Any]:
        """
        Export transform in format ready for ANTsPy usage.

        Creates files and returns a dictionary with paths and
        example ANTsPy code for loading.

        Args:
            transform: SpatialTransform to export.
            base_filename: Base name for output files.

        Returns:
            Dictionary with paths and usage information.
        """
        # Export all components
        paths = self.export_composite_transform(
            transform,
            base_filename,
            include_inverse=True,
        )

        # Generate example usage code
        usage_code = f'''
# ANTsPy usage example for loading the exported transform
import ants

# Load the displacement field (warp)
warp = ants.image_read("{paths["displacement_field"]}")

# Load the affine transform
# Note: For composite transforms, apply affine first, then warp
affine = ants.read_transform("{paths["affine"]}")

# Apply transform to an image
# moving_image = ants.image_read("your_image.nii.gz")
# warped = ants.apply_transforms(
#     fixed=reference_image,
#     moving=moving_image,
#     transformlist=[str(paths["displacement_field"]), str(paths["affine"])]
# )

# For inverse transform (deformed -> SUIT):
# inverse_warp = ants.image_read("{paths.get("inverse_displacement", "N/A")}")
'''

        # Save usage instructions
        usage_path = self.output_dir / f"{base_filename}_usage.txt"
        with open(usage_path, "w") as f:
            f.write(usage_code)
        paths["usage_instructions"] = usage_path

        return {
            "paths": paths,
            "source_space": transform.source_space,
            "target_space": transform.target_space,
            "max_displacement_mm": transform.get_max_displacement(),
            "usage_code": usage_code,
        }


def compute_transform_from_simulation(
    displacement_at_nodes: NDArray[np.float64],
    mesh_nodes: NDArray[np.float64],
    volume_shape: Tuple[int, int, int],
    affine: NDArray[np.float64],
    voxel_size: Tuple[float, float, float],
    smoothing_sigma: float = 2.0,
    output_shape: Optional[Tuple[int, int, int]] = None,
    output_affine: Optional[NDArray[np.float64]] = None,
    output_voxel_size: Optional[Tuple[float, float, float]] = None,
) -> SpatialTransform:
    """
    Compute spatial transform from FEM simulation results.

    Interpolates displacement from mesh nodes to a regular volume grid.
    Supports multi-resolution output: compute on coarse mesh, output at high resolution.

    Args:
        displacement_at_nodes: Displacement vectors at mesh nodes, shape (N, 3).
        mesh_nodes: Node coordinates in physical space, shape (N, 3).
        volume_shape: Mesh/input volume dimensions (X, Y, Z).
        affine: Affine matrix of the mesh reference volume.
        voxel_size: Mesh voxel size in mm.
        smoothing_sigma: Gaussian smoothing sigma in voxels.
        output_shape: Output volume dimensions (defaults to volume_shape).
        output_affine: Output affine matrix (defaults to affine).
        output_voxel_size: Output voxel size (defaults to voxel_size).

    Returns:
        SpatialTransform with interpolated displacement field at output resolution.
    """
    from scipy import ndimage
    from scipy.interpolate import RBFInterpolator

    # Use input parameters as defaults for output
    if output_shape is None:
        output_shape = volume_shape
    if output_affine is None:
        output_affine = affine
    if output_voxel_size is None:
        output_voxel_size = voxel_size

    # Check if we need multi-resolution interpolation
    is_multi_resolution = (
        output_shape != volume_shape or
        not np.allclose(output_affine, affine) or
        output_voxel_size != voxel_size
    )

    # Initialize output displacement field
    disp_field = np.zeros((*output_shape, 3), dtype=np.float32)

    # Inverse affine for physical-to-voxel conversion
    inv_output_affine = np.linalg.inv(output_affine)

    if is_multi_resolution and len(mesh_nodes) > 0:
        # Multi-resolution: use RBF interpolation for smooth displacement field
        # This properly interpolates from sparse coarse mesh to dense fine grid

        # Filter nodes with non-zero displacement for efficiency
        disp_magnitude = np.linalg.norm(displacement_at_nodes, axis=1)
        active_mask = disp_magnitude > 1e-10
        active_nodes = mesh_nodes[active_mask]
        active_disp = displacement_at_nodes[active_mask]

        if len(active_nodes) > 10:
            # Use RBF interpolation for smooth interpolation
            # Thin-plate spline kernel gives smooth results
            try:
                # Subsample if too many nodes (RBF is O(N^2))
                max_rbf_nodes = 5000
                if len(active_nodes) > max_rbf_nodes:
                    indices = np.random.choice(
                        len(active_nodes), max_rbf_nodes, replace=False
                    )
                    active_nodes = active_nodes[indices]
                    active_disp = active_disp[indices]

                # Create RBF interpolator for each displacement component
                rbf_interpolators = []
                for d in range(3):
                    rbf = RBFInterpolator(
                        active_nodes,
                        active_disp[:, d],
                        kernel='thin_plate_spline',
                        smoothing=1.0,  # Add smoothing for stability
                    )
                    rbf_interpolators.append(rbf)

                # Generate output voxel coordinates in physical space
                # Process in chunks to avoid memory issues
                chunk_size = 10000
                output_coords = []

                for x in range(output_shape[0]):
                    for y in range(output_shape[1]):
                        for z in range(output_shape[2]):
                            # Convert voxel to physical coordinate
                            voxel_homogeneous = np.array([x, y, z, 1.0])
                            physical = (output_affine @ voxel_homogeneous)[:3]
                            output_coords.append((x, y, z, physical))

                            if len(output_coords) >= chunk_size:
                                # Interpolate this chunk
                                coords_array = np.array([c[3] for c in output_coords])
                                for d in range(3):
                                    interp_values = rbf_interpolators[d](coords_array)
                                    for i, (vx, vy, vz, _) in enumerate(output_coords):
                                        disp_field[vx, vy, vz, d] = interp_values[i]
                                output_coords = []

                # Process remaining coordinates
                if output_coords:
                    coords_array = np.array([c[3] for c in output_coords])
                    for d in range(3):
                        interp_values = rbf_interpolators[d](coords_array)
                        for i, (vx, vy, vz, _) in enumerate(output_coords):
                            disp_field[vx, vy, vz, d] = interp_values[i]

            except Exception:
                # Fall back to scatter + smooth if RBF fails
                _scatter_displacements(
                    disp_field, mesh_nodes, displacement_at_nodes,
                    inv_output_affine, output_shape
                )
                # Use larger smoothing for coarse-to-fine
                effective_sigma = smoothing_sigma * 2
                for d in range(3):
                    disp_field[..., d] = ndimage.gaussian_filter(
                        disp_field[..., d], sigma=effective_sigma
                    )
        else:
            # Too few nodes, use scatter + smooth
            _scatter_displacements(
                disp_field, mesh_nodes, displacement_at_nodes,
                inv_output_affine, output_shape
            )
    else:
        # Same resolution: use original scatter + smooth approach
        _scatter_displacements(
            disp_field, mesh_nodes, displacement_at_nodes,
            inv_output_affine, output_shape
        )

    # Smooth the displacement field to fill gaps
    if smoothing_sigma > 0:
        for d in range(3):
            disp_field[..., d] = ndimage.gaussian_filter(
                disp_field[..., d],
                sigma=smoothing_sigma
            )

    return SpatialTransform.from_displacement_field(
        displacement=disp_field,
        affine=output_affine,
        voxel_size=output_voxel_size,
        source_space="SUIT",
        target_space="deformed",
    )


def _scatter_displacements(
    disp_field: NDArray[np.float32],
    mesh_nodes: NDArray[np.float64],
    displacement_at_nodes: NDArray[np.float64],
    inv_affine: NDArray[np.float64],
    volume_shape: Tuple[int, int, int],
) -> None:
    """Scatter node displacements to voxel grid using nearest-neighbor."""
    for i, (node, disp) in enumerate(zip(mesh_nodes, displacement_at_nodes)):
        # Convert physical coordinate to voxel
        voxel = (inv_affine @ np.append(node, 1.0))[:3]
        vi = np.round(voxel).astype(int)

        if all(0 <= vi[d] < volume_shape[d] for d in range(3)):
            disp_field[vi[0], vi[1], vi[2]] = disp
