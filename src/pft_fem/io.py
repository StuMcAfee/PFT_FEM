"""
Input/Output utilities for NIfTI format and other file operations.

Provides convenient wrappers around nibabel for reading and writing
medical imaging data in NIfTI format.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class NIfTIImage:
    """
    Container for NIfTI image data and metadata.

    Attributes:
        data: Image data array.
        affine: 4x4 affine transformation matrix.
        header: NIfTI header information.
        voxel_size: Voxel dimensions in mm.
    """

    data: NDArray
    affine: NDArray[np.float64]
    header: Optional[Any] = None
    voxel_size: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if self.voxel_size is None:
            self.voxel_size = tuple(np.abs(np.diag(self.affine)[:3]).tolist())


def load_nifti(filepath: Union[str, Path]) -> NIfTIImage:
    """
    Load a NIfTI image from file.

    Args:
        filepath: Path to NIfTI file (.nii or .nii.gz).

    Returns:
        NIfTIImage container with data and metadata.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is not a valid NIfTI.
    """
    import nibabel as nib

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    try:
        img = nib.load(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file: {e}")

    data = np.asarray(img.get_fdata())
    affine = img.affine.copy()

    return NIfTIImage(
        data=data,
        affine=affine,
        header=img.header.copy(),
    )


def save_nifti(
    data: NDArray,
    filepath: Union[str, Path],
    affine: Optional[NDArray[np.float64]] = None,
    header: Optional[Any] = None,
    dtype: Optional[np.dtype] = None,
) -> None:
    """
    Save data as a NIfTI file.

    Args:
        data: Image data array (3D or 4D).
        filepath: Output path (.nii or .nii.gz).
        affine: 4x4 affine matrix. If None, uses identity.
        header: NIfTI header. If None, creates new header.
        dtype: Output data type. If None, uses input dtype.
    """
    import nibabel as nib

    filepath = Path(filepath)

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Default affine (identity with 1mm voxels)
    if affine is None:
        affine = np.eye(4)

    # Convert data type if specified
    if dtype is not None:
        data = data.astype(dtype)

    # Create NIfTI image
    if header is not None:
        img = nib.Nifti1Image(data, affine, header=header)
    else:
        img = nib.Nifti1Image(data, affine)

    # Save
    nib.save(img, filepath)


class NIfTIWriter:
    """
    Utility class for writing simulation results to NIfTI format.

    Provides methods to save various simulation outputs with
    appropriate metadata and naming conventions.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        affine: Optional[NDArray[np.float64]] = None,
        base_name: str = "simulation",
    ):
        """
        Initialize NIfTI writer.

        Args:
            output_dir: Directory for output files.
            affine: Default affine transformation matrix.
            base_name: Base name for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.affine = affine if affine is not None else np.eye(4)
        self.base_name = base_name

    def write_volume(
        self,
        data: NDArray,
        name: str,
        description: Optional[str] = None,
        affine: Optional[NDArray[np.float64]] = None,
    ) -> Path:
        """
        Write a single volume to NIfTI file.

        Args:
            data: 3D volume data.
            name: Name suffix for the file.
            description: Description stored in header.
            affine: Override affine matrix.

        Returns:
            Path to written file.
        """
        import nibabel as nib

        filename = f"{self.base_name}_{name}.nii.gz"
        filepath = self.output_dir / filename

        use_affine = affine if affine is not None else self.affine

        img = nib.Nifti1Image(data.astype(np.float32), use_affine)

        if description:
            img.header["descrip"] = description.encode()[:80]

        nib.save(img, filepath)

        return filepath

    def write_label_volume(
        self,
        data: NDArray[np.int32],
        name: str,
        label_names: Optional[Dict[int, str]] = None,
    ) -> Path:
        """
        Write a label/segmentation volume.

        Args:
            data: 3D integer label volume.
            name: Name suffix for the file.
            label_names: Mapping of label IDs to names.

        Returns:
            Path to written file.
        """
        filename = f"{self.base_name}_{name}.nii.gz"
        filepath = self.output_dir / filename

        save_nifti(data.astype(np.int16), filepath, self.affine)

        # Optionally save label names as JSON sidecar
        if label_names:
            import json
            json_path = self.output_dir / f"{self.base_name}_{name}_labels.json"
            with open(json_path, "w") as f:
                json.dump(label_names, f, indent=2)

        return filepath

    def write_mask(
        self,
        mask: NDArray[np.bool_],
        name: str,
    ) -> Path:
        """
        Write a binary mask volume.

        Args:
            mask: 3D boolean mask.
            name: Name suffix for the file.

        Returns:
            Path to written file.
        """
        return self.write_volume(
            mask.astype(np.uint8),
            name,
            description=f"Binary mask: {name}"
        )

    def write_time_series(
        self,
        volumes: list,
        name: str,
        time_points: Optional[list] = None,
    ) -> Path:
        """
        Write a 4D time series volume.

        Args:
            volumes: List of 3D volumes.
            name: Name suffix for the file.
            time_points: Time values for each volume.

        Returns:
            Path to written file.
        """
        import nibabel as nib

        # Stack into 4D
        data_4d = np.stack(volumes, axis=-1).astype(np.float32)

        filename = f"{self.base_name}_{name}_4d.nii.gz"
        filepath = self.output_dir / filename

        img = nib.Nifti1Image(data_4d, self.affine)

        # Store time information in header
        img.header.set_xyzt_units("mm", "sec")
        if time_points:
            # Time step
            dt = time_points[1] - time_points[0] if len(time_points) > 1 else 1.0
            img.header["pixdim"][4] = dt

        nib.save(img, filepath)

        return filepath

    def write_displacement_field(
        self,
        displacement: NDArray[np.float64],
        name: str = "displacement",
    ) -> Path:
        """
        Write a 3D displacement field (for deformation visualization).

        Args:
            displacement: Displacement field, shape (X, Y, Z, 3).
            name: Name suffix.

        Returns:
            Path to written file.
        """
        import nibabel as nib

        filename = f"{self.base_name}_{name}.nii.gz"
        filepath = self.output_dir / filename

        # NIfTI stores vector fields with vector dimension as 5th dimension
        data = displacement.astype(np.float32)

        img = nib.Nifti1Image(data, self.affine)
        img.header.set_intent("vector", name="displacement")

        nib.save(img, filepath)

        return filepath

    def write_spatial_transform(
        self,
        transform: "SpatialTransform",
        base_name: str = "spatial_transform",
        include_inverse: bool = True,
    ) -> Dict[str, Path]:
        """
        Write spatial transform in ANTsPy-compatible format.

        Exports the complete transform including:
        - Displacement field (NIfTI, ANTs-compatible)
        - Affine transform (.mat file)
        - Inverse displacement field (optional)
        - Jacobian determinant map
        - Metadata (JSON)

        Args:
            transform: SpatialTransform to export.
            base_name: Base name for output files.
            include_inverse: Whether to export inverse transform.

        Returns:
            Dictionary mapping output type to file path.
        """
        from .transforms import ANTsTransformExporter

        exporter = ANTsTransformExporter(self.output_dir)
        return exporter.export_composite_transform(
            transform,
            base_filename=f"{self.base_name}_{base_name}",
            include_inverse=include_inverse,
        )

    def write_simulation_results(
        self,
        result: "SimulationResult",
        export_transform: bool = True,
    ) -> Dict[str, Path]:
        """
        Write all simulation results to NIfTI files.

        Args:
            result: SimulationResult from MRISimulator.
            export_transform: Whether to export the spatial transform.

        Returns:
            Dictionary mapping output type to file path.
        """
        from .simulation import SimulationResult

        paths = {}

        # Write MRI images
        for seq_name, volume in result.mri_images.items():
            paths[f"mri_{seq_name}"] = self.write_volume(
                volume, f"mri_{seq_name}",
                description=f"Simulated {seq_name} MRI"
            )

        # Write deformed atlas
        paths["deformed_atlas"] = self.write_volume(
            result.deformed_atlas,
            "deformed_atlas",
            description="Atlas with tumor deformation"
        )

        # Write masks
        paths["tumor_mask"] = self.write_mask(result.tumor_mask, "tumor_mask")
        paths["edema_mask"] = self.write_mask(result.edema_mask, "edema_mask")

        # Export spatial transform (SUIT -> deformed) in ANTsPy format
        if export_transform and result.spatial_transform is not None:
            transform_paths = self.write_spatial_transform(
                result.spatial_transform,
                base_name="suit_to_deformed",
                include_inverse=True,
            )
            paths.update({f"transform_{k}": v for k, v in transform_paths.items()})

        # Write tumor density evolution
        if len(result.tumor_states) > 1:
            # Interpolate tumor density to volumes
            densities = []
            for state in result.tumor_states[::5]:  # Every 5th time point
                # This would need mesh-to-volume interpolation
                # For now, just note this as a TODO
                pass

        # Save metadata as JSON
        import json
        meta_path = self.output_dir / f"{self.base_name}_metadata.json"

        # Convert numpy types for JSON serialization
        serializable_meta = {}
        for key, value in result.metadata.items():
            if isinstance(value, np.ndarray):
                serializable_meta[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                serializable_meta[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                serializable_meta[key] = float(value)
            elif isinstance(value, tuple):
                serializable_meta[key] = list(value)
            elif isinstance(value, dict):
                # Handle nested dicts (like spatial_transform_info)
                serializable_meta[key] = self._serialize_dict(value)
            else:
                serializable_meta[key] = value

        with open(meta_path, "w") as f:
            json.dump(serializable_meta, f, indent=2)

        paths["metadata"] = meta_path

        return paths

    def _serialize_dict(self, d: Dict) -> Dict:
        """Recursively serialize dictionary values for JSON."""
        result = {}
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                result[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                result[key] = float(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            elif isinstance(value, dict):
                result[key] = self._serialize_dict(value)
            else:
                result[key] = value
        return result


def validate_nifti(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a NIfTI file and return information about it.

    Args:
        filepath: Path to NIfTI file.

    Returns:
        Dictionary with file information and any issues found.
    """
    import nibabel as nib

    filepath = Path(filepath)
    info = {
        "path": str(filepath),
        "exists": filepath.exists(),
        "valid": False,
        "issues": [],
    }

    if not info["exists"]:
        info["issues"].append("File does not exist")
        return info

    try:
        img = nib.load(filepath)
        data = img.get_fdata()

        info["valid"] = True
        info["shape"] = data.shape
        info["dtype"] = str(data.dtype)
        info["voxel_size"] = tuple(img.header.get_zooms()[:3])
        info["affine_det"] = float(np.linalg.det(img.affine[:3, :3]))
        info["data_range"] = (float(np.min(data)), float(np.max(data)))

        # Check for common issues
        if info["affine_det"] < 0:
            info["issues"].append("Negative affine determinant (left-handed)")

        if np.any(np.isnan(data)):
            info["issues"].append("Contains NaN values")

        if np.any(np.isinf(data)):
            info["issues"].append("Contains infinite values")

        if len(data.shape) not in [3, 4]:
            info["issues"].append(f"Unusual dimensionality: {len(data.shape)}")

    except Exception as e:
        info["issues"].append(f"Failed to load: {str(e)}")

    return info


def resample_nifti(
    source: Union[str, Path, NIfTIImage],
    target_shape: Tuple[int, int, int],
    target_affine: Optional[NDArray[np.float64]] = None,
    order: int = 1,
) -> NIfTIImage:
    """
    Resample a NIfTI image to a new shape/resolution.

    Args:
        source: Source NIfTI file or image.
        target_shape: Target volume dimensions.
        target_affine: Target affine matrix.
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        Resampled NIfTIImage.
    """
    from scipy import ndimage

    if isinstance(source, (str, Path)):
        source = load_nifti(source)

    # Compute zoom factors
    zoom_factors = np.array(target_shape) / np.array(source.data.shape[:3])

    # Resample
    if source.data.ndim == 3:
        resampled = ndimage.zoom(source.data, zoom_factors, order=order)
    else:
        # 4D: resample spatial dimensions only
        resampled = np.stack([
            ndimage.zoom(source.data[..., t], zoom_factors, order=order)
            for t in range(source.data.shape[3])
        ], axis=-1)

    # Update affine
    if target_affine is None:
        target_affine = source.affine.copy()
        target_affine[:3, :3] /= zoom_factors[:, np.newaxis]

    return NIfTIImage(
        data=resampled,
        affine=target_affine,
    )


def combine_masks(
    masks: Dict[str, NDArray[np.bool_]],
    output_path: Optional[Union[str, Path]] = None,
    affine: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.int32]:
    """
    Combine multiple binary masks into a single label volume.

    Args:
        masks: Dictionary mapping label names to binary masks.
        output_path: Optional path to save combined label volume.
        affine: Affine matrix for output (if saving).

    Returns:
        Combined label volume with unique integer labels.
    """
    # Get shape from first mask
    first_mask = next(iter(masks.values()))
    combined = np.zeros(first_mask.shape, dtype=np.int32)

    label_map = {}
    for i, (name, mask) in enumerate(masks.items(), start=1):
        combined[mask] = i
        label_map[i] = name

    if output_path is not None:
        if affine is None:
            affine = np.eye(4)
        save_nifti(combined, output_path, affine)

        # Save label map
        import json
        json_path = Path(output_path).with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(label_map, f, indent=2)

    return combined
