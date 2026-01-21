"""
SUIT Atlas loading and processing module.

The SUIT (Spatially Unbiased Infratentorial Template) atlas provides
anatomical templates for the cerebellum and brainstem, which form the
posterior cranial fossa region.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class AtlasRegion:
    """Represents a labeled region in the atlas."""

    label_id: int
    name: str
    mask: NDArray[np.bool_]
    volume_mm3: float
    centroid: Tuple[float, float, float]


@dataclass
class AtlasData:
    """Container for loaded atlas data."""

    template: NDArray[np.float32]
    labels: NDArray[np.int32]
    affine: NDArray[np.float64]
    voxel_size: Tuple[float, float, float]
    shape: Tuple[int, int, int]
    regions: Dict[int, AtlasRegion]


class SUITAtlasLoader:
    """
    Loader for SUIT cerebellar atlas data.

    The SUIT atlas provides:
    - T1-weighted template image
    - Probabilistic maps for cerebellar lobules
    - Tissue segmentation masks

    This loader can work with standard NIfTI atlas files or generate
    synthetic atlas data for testing purposes.
    """

    # Standard SUIT atlas region labels for the posterior fossa
    REGION_LABELS = {
        1: "Left Cerebellum I-IV",
        2: "Right Cerebellum I-IV",
        3: "Left Cerebellum V",
        4: "Right Cerebellum V",
        5: "Left Cerebellum VI",
        6: "Right Cerebellum VI",
        7: "Vermis VI",
        8: "Left Cerebellum Crus I",
        9: "Right Cerebellum Crus I",
        10: "Left Cerebellum Crus II",
        11: "Right Cerebellum Crus II",
        12: "Left Cerebellum VIIb",
        13: "Right Cerebellum VIIb",
        14: "Left Cerebellum VIIIa",
        15: "Right Cerebellum VIIIa",
        16: "Left Cerebellum VIIIb",
        17: "Right Cerebellum VIIIb",
        18: "Left Cerebellum IX",
        19: "Right Cerebellum IX",
        20: "Left Cerebellum X",
        21: "Right Cerebellum X",
        22: "Vermis Crus I",
        23: "Vermis Crus II",
        24: "Vermis VIIb",
        25: "Vermis VIIIa",
        26: "Vermis VIIIb",
        27: "Vermis IX",
        28: "Vermis X",
        29: "Brainstem",
        30: "Fourth Ventricle",
    }

    def __init__(self, atlas_dir: Optional[Path] = None):
        """
        Initialize the SUIT atlas loader.

        Args:
            atlas_dir: Path to directory containing SUIT atlas files.
                      If None, synthetic data will be generated.
        """
        self.atlas_dir = Path(atlas_dir) if atlas_dir else None
        self._cached_data: Optional[AtlasData] = None

    def load(self, use_cache: bool = True) -> AtlasData:
        """
        Load the SUIT atlas data.

        Args:
            use_cache: If True, return cached data if available.

        Returns:
            AtlasData containing template, labels, and region information.
        """
        if use_cache and self._cached_data is not None:
            return self._cached_data

        if self.atlas_dir is not None and self.atlas_dir.exists():
            data = self._load_from_files()
        else:
            data = self._generate_synthetic_atlas()

        if use_cache:
            self._cached_data = data

        return data

    def _load_from_files(self) -> AtlasData:
        """Load atlas from NIfTI files."""
        import nibabel as nib

        template_path = self.atlas_dir / "SUIT_template.nii.gz"
        labels_path = self.atlas_dir / "SUIT_labels.nii.gz"

        if not template_path.exists():
            template_path = self.atlas_dir / "SUIT_template.nii"
        if not labels_path.exists():
            labels_path = self.atlas_dir / "SUIT_labels.nii"

        template_img = nib.load(template_path)
        template_data = np.asarray(template_img.get_fdata(), dtype=np.float32)
        affine = template_img.affine

        labels_img = nib.load(labels_path)
        labels_data = np.asarray(labels_img.get_fdata(), dtype=np.int32)

        voxel_size = tuple(np.abs(np.diag(affine)[:3]).tolist())

        regions = self._extract_regions(labels_data, voxel_size)

        return AtlasData(
            template=template_data,
            labels=labels_data,
            affine=affine,
            voxel_size=voxel_size,
            shape=template_data.shape,
            regions=regions,
        )

    def _generate_synthetic_atlas(
        self,
        shape: Tuple[int, int, int] = (91, 109, 91),
        voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    ) -> AtlasData:
        """
        Generate synthetic atlas data for testing.

        Creates a simplified posterior fossa representation with:
        - Cerebellar hemispheres (left and right)
        - Vermis (midline)
        - Brainstem
        - Fourth ventricle

        Args:
            shape: Volume dimensions in voxels.
            voxel_size: Voxel dimensions in mm.

        Returns:
            Synthetic AtlasData.
        """
        template = np.zeros(shape, dtype=np.float32)
        labels = np.zeros(shape, dtype=np.int32)

        # Create coordinate grids
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        center = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 3])

        # Posterior fossa is in the lower posterior part of the brain
        # Model cerebellum as two ellipsoids (hemispheres) + vermis

        # Left cerebellar hemisphere
        left_center = center + np.array([-15, 0, 0])
        left_dist = ((x - left_center[0]) / 20) ** 2 + \
                    ((y - left_center[1]) / 25) ** 2 + \
                    ((z - left_center[2]) / 15) ** 2
        left_mask = left_dist < 1

        # Right cerebellar hemisphere
        right_center = center + np.array([15, 0, 0])
        right_dist = ((x - right_center[0]) / 20) ** 2 + \
                     ((y - right_center[1]) / 25) ** 2 + \
                     ((z - right_center[2]) / 15) ** 2
        right_mask = right_dist < 1

        # Vermis (midline structure)
        vermis_center = center
        vermis_dist = ((x - vermis_center[0]) / 8) ** 2 + \
                      ((y - vermis_center[1]) / 20) ** 2 + \
                      ((z - vermis_center[2]) / 12) ** 2
        vermis_mask = vermis_dist < 1

        # Brainstem (cylindrical, anterior to cerebellum)
        brainstem_center = center + np.array([0, -20, 5])
        brainstem_dist = ((x - brainstem_center[0]) / 8) ** 2 + \
                         ((y - brainstem_center[1]) / 8) ** 2
        brainstem_mask = (brainstem_dist < 1) & (z > center[2] - 20) & (z < center[2] + 25)

        # Fourth ventricle (small cavity between cerebellum and brainstem)
        ventricle_center = center + np.array([0, -10, 0])
        ventricle_dist = ((x - ventricle_center[0]) / 4) ** 2 + \
                         ((y - ventricle_center[1]) / 3) ** 2 + \
                         ((z - ventricle_center[2]) / 6) ** 2
        ventricle_mask = ventricle_dist < 1

        # Assign labels (simplified - using key regions)
        labels[left_mask] = 5  # Left Cerebellum VI (representative)
        labels[right_mask] = 6  # Right Cerebellum VI
        labels[vermis_mask] = 7  # Vermis VI
        labels[brainstem_mask] = 29  # Brainstem
        labels[ventricle_mask] = 30  # Fourth Ventricle

        # Create template intensities
        # Gray matter (cerebellum) ~100, White matter ~150, CSF (ventricle) ~30
        template[left_mask | right_mask | vermis_mask] = 100.0
        template[brainstem_mask] = 120.0
        template[ventricle_mask] = 30.0

        # Add some noise for realism
        noise = np.random.normal(0, 5, shape).astype(np.float32)
        template = np.clip(template + noise * (template > 0), 0, 255)

        # Create affine matrix
        affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
        affine[:3, 3] = -np.array(shape) * np.array(voxel_size) / 2

        regions = self._extract_regions(labels, voxel_size)

        return AtlasData(
            template=template,
            labels=labels,
            affine=affine,
            voxel_size=voxel_size,
            shape=shape,
            regions=regions,
        )

    def _extract_regions(
        self,
        labels: NDArray[np.int32],
        voxel_size: Tuple[float, float, float],
    ) -> Dict[int, AtlasRegion]:
        """Extract region information from label volume."""
        regions = {}
        voxel_volume = np.prod(voxel_size)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background

        for label_id in unique_labels:
            mask = labels == label_id
            voxel_count = np.sum(mask)

            if voxel_count == 0:
                continue

            # Calculate centroid
            coords = np.array(np.where(mask))
            centroid = tuple(np.mean(coords, axis=1).tolist())

            # Get region name
            name = self.REGION_LABELS.get(int(label_id), f"Region {label_id}")

            regions[int(label_id)] = AtlasRegion(
                label_id=int(label_id),
                name=name,
                mask=mask,
                volume_mm3=float(voxel_count * voxel_volume),
                centroid=centroid,
            )

        return regions

    def get_region_by_name(self, name: str) -> Optional[int]:
        """Get region label ID by name (partial match)."""
        name_lower = name.lower()
        for label_id, region_name in self.REGION_LABELS.items():
            if name_lower in region_name.lower():
                return label_id
        return None


class AtlasProcessor:
    """
    Processor for atlas data manipulation and analysis.

    Provides utilities for:
    - Extracting tissue masks
    - Computing distance fields
    - Resampling and transforming atlas data
    - Preparing data for mesh generation
    """

    def __init__(self, atlas_data: AtlasData):
        """
        Initialize processor with atlas data.

        Args:
            atlas_data: Loaded atlas data from SUITAtlasLoader.
        """
        self.atlas = atlas_data

    def get_tissue_mask(
        self,
        tissue_type: str = "cerebellum",
    ) -> NDArray[np.bool_]:
        """
        Extract a binary mask for a tissue type.

        Args:
            tissue_type: One of "cerebellum", "brainstem", "ventricle", "all".

        Returns:
            Binary mask array.
        """
        labels = self.atlas.labels

        if tissue_type == "cerebellum":
            # Labels 1-28 are cerebellar structures
            mask = (labels >= 1) & (labels <= 28)
        elif tissue_type == "brainstem":
            mask = labels == 29
        elif tissue_type == "ventricle":
            mask = labels == 30
        elif tissue_type == "all":
            mask = labels > 0
        else:
            raise ValueError(f"Unknown tissue type: {tissue_type}")

        return mask

    def compute_distance_field(
        self,
        mask: Optional[NDArray[np.bool_]] = None,
        signed: bool = True,
    ) -> NDArray[np.float32]:
        """
        Compute distance field from tissue boundary.

        Args:
            mask: Binary mask to compute distance from. If None, uses all tissue.
            signed: If True, negative values inside, positive outside.

        Returns:
            Distance field in mm.
        """
        from scipy import ndimage

        if mask is None:
            mask = self.get_tissue_mask("all")

        # Compute unsigned distance transform
        dist_outside = ndimage.distance_transform_edt(
            ~mask, sampling=self.atlas.voxel_size
        )

        if signed:
            dist_inside = ndimage.distance_transform_edt(
                mask, sampling=self.atlas.voxel_size
            )
            return (dist_outside - dist_inside).astype(np.float32)

        return dist_outside.astype(np.float32)

    def extract_surface_points(
        self,
        mask: Optional[NDArray[np.bool_]] = None,
        spacing: int = 2,
    ) -> NDArray[np.float64]:
        """
        Extract surface points from a mask for mesh generation.

        Args:
            mask: Binary mask. If None, uses all tissue.
            spacing: Subsampling factor for surface points.

        Returns:
            Array of shape (N, 3) with surface point coordinates in mm.
        """
        from scipy import ndimage

        if mask is None:
            mask = self.get_tissue_mask("all")

        # Find surface voxels (boundary between mask and non-mask)
        eroded = ndimage.binary_erosion(mask)
        surface = mask & ~eroded

        # Get coordinates
        coords = np.array(np.where(surface)).T

        # Subsample if needed
        if spacing > 1:
            coords = coords[::spacing]

        # Convert to physical coordinates (mm)
        physical_coords = coords * np.array(self.atlas.voxel_size)

        # Apply affine offset
        physical_coords += self.atlas.affine[:3, 3]

        return physical_coords

    def resample(
        self,
        target_shape: Tuple[int, int, int],
        order: int = 1,
    ) -> AtlasData:
        """
        Resample atlas to a new resolution.

        Args:
            target_shape: Target volume dimensions.
            order: Interpolation order (0=nearest, 1=linear, 3=cubic).

        Returns:
            Resampled AtlasData.
        """
        from scipy import ndimage

        zoom_factors = np.array(target_shape) / np.array(self.atlas.shape)

        # Resample template with interpolation
        new_template = ndimage.zoom(
            self.atlas.template, zoom_factors, order=order
        ).astype(np.float32)

        # Resample labels with nearest neighbor
        new_labels = ndimage.zoom(
            self.atlas.labels, zoom_factors, order=0
        ).astype(np.int32)

        # Update voxel size and affine
        new_voxel_size = tuple(
            (v / z for v, z in zip(self.atlas.voxel_size, zoom_factors))
        )

        new_affine = self.atlas.affine.copy()
        new_affine[:3, :3] = np.diag(new_voxel_size)

        # Re-extract regions
        loader = SUITAtlasLoader()
        regions = loader._extract_regions(new_labels, new_voxel_size)

        return AtlasData(
            template=new_template,
            labels=new_labels,
            affine=new_affine,
            voxel_size=new_voxel_size,
            shape=target_shape,
            regions=regions,
        )

    def get_bounding_box(
        self,
        mask: Optional[NDArray[np.bool_]] = None,
        padding: int = 5,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get bounding box of tissue region.

        Args:
            mask: Binary mask. If None, uses all tissue.
            padding: Padding voxels around the bounding box.

        Returns:
            Tuple of (min_coords, max_coords).
        """
        if mask is None:
            mask = self.get_tissue_mask("all")

        coords = np.array(np.where(mask))

        min_coords = np.maximum(coords.min(axis=1) - padding, 0)
        max_coords = np.minimum(
            coords.max(axis=1) + padding + 1,
            np.array(self.atlas.shape)
        )

        return tuple(min_coords.tolist()), tuple(max_coords.tolist())

    def crop_to_region(
        self,
        padding: int = 5,
    ) -> AtlasData:
        """
        Crop atlas to bounding box of tissue.

        Args:
            padding: Padding voxels around the tissue.

        Returns:
            Cropped AtlasData.
        """
        min_c, max_c = self.get_bounding_box(padding=padding)

        slices = tuple(slice(mi, ma) for mi, ma in zip(min_c, max_c))

        new_template = self.atlas.template[slices].copy()
        new_labels = self.atlas.labels[slices].copy()
        new_shape = new_template.shape

        # Update affine to reflect new origin
        new_affine = self.atlas.affine.copy()
        offset = np.array(min_c) * np.array(self.atlas.voxel_size)
        new_affine[:3, 3] += offset

        # Re-extract regions
        loader = SUITAtlasLoader()
        regions = loader._extract_regions(new_labels, self.atlas.voxel_size)

        return AtlasData(
            template=new_template,
            labels=new_labels,
            affine=new_affine,
            voxel_size=self.atlas.voxel_size,
            shape=new_shape,
            regions=regions,
        )
