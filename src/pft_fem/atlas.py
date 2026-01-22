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

    # Standard SUIT atlas region labels (from Diedrichsen 2009 atlas)
    # Labels match atl-Anatom_space-SUIT_dseg.nii
    REGION_LABELS = {
        1: "Left I-IV",
        2: "Right I-IV",
        3: "Left V",
        4: "Right V",
        5: "Left VI",
        6: "Vermis VI",
        7: "Right VI",
        8: "Left Crus I",
        9: "Vermis Crus I",
        10: "Right Crus I",
        11: "Left Crus II",
        12: "Vermis Crus II",
        13: "Right Crus II",
        14: "Left VIIb",
        15: "Vermis VIIb",
        16: "Right VIIb",
        17: "Left VIIIa",
        18: "Vermis VIIIa",
        19: "Right VIIIa",
        20: "Left VIIIb",
        21: "Vermis VIIIb",
        22: "Right VIIIb",
        23: "Left IX",
        24: "Vermis IX",
        25: "Right IX",
        26: "Left X",
        27: "Vermis X",
        28: "Right X",
        29: "Left Dentate",
        30: "Right Dentate",
        31: "Left Interposed",
        32: "Right Interposed",
        33: "Left Fastigial",
        34: "Right Fastigial",
    }

    # Default path to bundled atlas files
    DEFAULT_ATLAS_DIR = Path(__file__).parent.parent.parent / "data" / "atlases" / "SUIT"

    def __init__(self, atlas_dir: Optional[Path] = None, use_bundled: bool = True):
        """
        Initialize the SUIT atlas loader.

        Args:
            atlas_dir: Path to directory containing SUIT atlas files.
                      If None and use_bundled=True, uses bundled atlas files.
                      If None and use_bundled=False, generates synthetic data.
            use_bundled: If True and atlas_dir is None, use bundled atlas files
                        from DiedrichsenLab/cerebellar_atlases repository.
        """
        if atlas_dir is not None:
            self.atlas_dir = Path(atlas_dir)
        elif use_bundled and self.DEFAULT_ATLAS_DIR.exists():
            self.atlas_dir = self.DEFAULT_ATLAS_DIR
        else:
            self.atlas_dir = None
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

        # Try bundled DiedrichsenLab atlas filenames first
        template_path = self.atlas_dir / "tpl-SUIT_T1w.nii"
        labels_path = self.atlas_dir / "atl-Anatom_space-SUIT_dseg.nii"

        # Fall back to legacy filenames if not found
        if not template_path.exists():
            template_path = self.atlas_dir / "SUIT_template.nii.gz"
            if not template_path.exists():
                template_path = self.atlas_dir / "SUIT_template.nii"

        if not labels_path.exists():
            labels_path = self.atlas_dir / "SUIT_labels.nii.gz"
            if not labels_path.exists():
                labels_path = self.atlas_dir / "SUIT_labels.nii"

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found in {self.atlas_dir}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found in {self.atlas_dir}")

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
            tissue_type: One of "cerebellum", "lobules", "nuclei", "all".

        Returns:
            Binary mask array.
        """
        labels = self.atlas.labels

        if tissue_type == "cerebellum" or tissue_type == "all":
            # All cerebellar structures (lobules + nuclei)
            mask = labels > 0
        elif tissue_type == "lobules":
            # Labels 1-28 are cerebellar lobules
            mask = (labels >= 1) & (labels <= 28)
        elif tissue_type == "nuclei":
            # Labels 29-34 are deep cerebellar nuclei
            mask = (labels >= 29) & (labels <= 34)
        elif tissue_type == "brainstem":
            # Not in standard SUIT atlas, return empty mask
            mask = np.zeros_like(labels, dtype=bool)
        elif tissue_type == "ventricle":
            # Not in standard SUIT atlas, return empty mask
            mask = np.zeros_like(labels, dtype=bool)
        elif tissue_type == "template" or tissue_type == "anatomical":
            # Use template image to get all tissue (includes brainstem)
            mask = self.get_anatomical_mask()
        else:
            raise ValueError(f"Unknown tissue type: {tissue_type}")

        return mask

    def get_anatomical_mask(
        self,
        threshold: Optional[float] = None,
        include_brainstem: bool = True,
    ) -> NDArray[np.bool_]:
        """
        Create a tissue mask from the anatomical template image.

        This method thresholds the T1 template image to create a mask that
        includes all visible tissue, not just the labeled cerebellar regions.
        This is important for including the brainstem in tumor simulations.

        Args:
            threshold: Intensity threshold for the template. If None, uses
                      Otsu's method to automatically determine threshold.
            include_brainstem: If True (default), includes all tissue visible
                              in the template. If False, uses only labeled regions.

        Returns:
            Binary mask of all tissue in the template image.
        """
        if not include_brainstem:
            # Fall back to label-based mask
            return self.atlas.labels > 0

        template = self.atlas.template

        if threshold is None:
            # Use Otsu's method for automatic thresholding
            # Only consider non-zero voxels
            nonzero_values = template[template > 0]
            if len(nonzero_values) == 0:
                return np.zeros_like(template, dtype=bool)

            # Simple Otsu implementation
            hist, bin_edges = np.histogram(nonzero_values, bins=256)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Compute cumulative sums and means
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]

            mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + 1e-10))[::-1]

            # Compute between-class variance
            variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

            # Find optimal threshold
            idx = np.argmax(variance)
            threshold = bin_centers[idx]

        # Create mask from thresholded template
        mask = template > threshold

        # Clean up with morphological operations
        from scipy import ndimage

        # Remove small isolated regions
        mask = ndimage.binary_opening(mask, iterations=1)

        # Fill small holes
        mask = ndimage.binary_closing(mask, iterations=1)

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
