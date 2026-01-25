"""
Biophysical constraints module for realistic brain tissue modeling.

This module provides:
- Tissue segmentation (white/gray matter, CSF) from ICBM-152 MNI atlases
- Fiber orientation data from JHU DTI atlas
- Posterior fossa region masking (cerebellum + brainstem)
- Anisotropic material properties based on tissue microstructure
- Optional SUIT space transformation for cerebellar-focused analysis

Default configuration:
- Template: ICBM-152 in MNI space
- Tumor origin: MNI coordinates [2, -49, -35] (vermis/fourth ventricle region)
- Modeled region: Posterior fossa (cerebellum + brainstem only)

References:
- ICBM-152: Fonov et al., NeuroImage 2011
- JHU DTI Atlas: Mori et al., MRI Atlas of Human White Matter, 2005
- SUIT Atlas: Diedrichsen et al., NeuroImage 2006
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union, Any
from enum import Enum
import warnings

import numpy as np
from numpy.typing import NDArray


# Default tumor origin in MNI coordinates (vermis/fourth ventricle region)
DEFAULT_TUMOR_ORIGIN_MNI = np.array([2.0, -49.0, -35.0])

# Posterior fossa bounding box in MNI coordinates (approximate)
# Covers cerebellum and brainstem
POSTERIOR_FOSSA_BOUNDS_MNI = {
    'x_min': -55.0, 'x_max': 55.0,   # Left-right (symmetric)
    'y_min': -100.0, 'y_max': -20.0,  # Anterior-posterior (posterior region)
    'z_min': -70.0, 'z_max': 10.0,    # Superior-inferior (inferior region)
}


class BrainTissue(Enum):
    """Brain tissue types with distinct mechanical properties."""

    BACKGROUND = 0
    CSF = 1
    GRAY_MATTER = 2
    WHITE_MATTER = 3
    SKULL = 4
    SCALP = 5

    @classmethod
    def from_label(cls, label: int) -> "BrainTissue":
        """Convert numeric label to tissue type."""
        mapping = {0: cls.BACKGROUND, 1: cls.CSF, 2: cls.GRAY_MATTER,
                   3: cls.WHITE_MATTER, 4: cls.SKULL, 5: cls.SCALP}
        return mapping.get(label, cls.BACKGROUND)


@dataclass
class FiberOrientation:
    """
    Fiber orientation data at each voxel/node.

    Stores the principal diffusion direction from DTI, which indicates
    the local white matter fiber orientation.

    Attributes:
        vectors: Principal fiber direction vectors (N, 3), unit normalized
        fractional_anisotropy: FA values (0-1) indicating fiber coherence
        mean_diffusivity: MD values for diffusion magnitude
        affine: Coordinate transformation matrix
    """

    vectors: NDArray[np.float64]  # (N, 3) or volume shape + (3,)
    fractional_anisotropy: NDArray[np.float32]  # Scalar field
    mean_diffusivity: Optional[NDArray[np.float32]] = None
    affine: Optional[NDArray[np.float64]] = None

    def get_direction_at_point(
        self,
        point: NDArray[np.float64],
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> NDArray[np.float64]:
        """Get interpolated fiber direction at a physical coordinate."""
        if self.affine is not None:
            # Convert physical to voxel coordinates
            inv_affine = np.linalg.inv(self.affine)
            point_h = np.append(point, 1.0)
            voxel_coord = (inv_affine @ point_h)[:3]
        else:
            voxel_coord = point / np.array(voxel_size)

        # Trilinear interpolation
        return self._trilinear_interpolate_vector(voxel_coord)

    def _trilinear_interpolate_vector(
        self,
        coord: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Trilinear interpolation of vector field."""
        if self.vectors.ndim == 2:
            # Already a list of vectors, find nearest
            idx = int(np.round(coord[0]))
            idx = max(0, min(idx, len(self.vectors) - 1))
            return self.vectors[idx]

        # Volume data
        shape = self.vectors.shape[:3]
        x, y, z = coord

        # Clamp to valid range
        x = max(0, min(x, shape[0] - 1.001))
        y = max(0, min(y, shape[1] - 1.001))
        z = max(0, min(z, shape[2] - 1.001))

        x0, y0, z0 = int(x), int(y), int(z)
        x1 = min(x0 + 1, shape[0] - 1)
        y1 = min(y0 + 1, shape[1] - 1)
        z1 = min(z0 + 1, shape[2] - 1)

        xd, yd, zd = x - x0, y - y0, z - z0

        # Interpolate
        c000 = self.vectors[x0, y0, z0]
        c001 = self.vectors[x0, y0, z1]
        c010 = self.vectors[x0, y1, z0]
        c011 = self.vectors[x0, y1, z1]
        c100 = self.vectors[x1, y0, z0]
        c101 = self.vectors[x1, y0, z1]
        c110 = self.vectors[x1, y1, z0]
        c111 = self.vectors[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        result = c0 * (1 - zd) + c1 * zd

        # Normalize
        norm = np.linalg.norm(result)
        if norm > 1e-6:
            result = result / norm
        else:
            result = np.array([1.0, 0.0, 0.0])

        return result


@dataclass
class TissueSegmentation:
    """
    Brain tissue segmentation data.

    Provides voxel-wise tissue classification and probability maps
    for white matter, gray matter, and CSF.

    Attributes:
        labels: Discrete tissue labels at each voxel
        wm_probability: White matter probability map (0-1)
        gm_probability: Gray matter probability map (0-1)
        csf_probability: CSF probability map (0-1)
        affine: Coordinate transformation matrix
        voxel_size: Physical voxel dimensions in mm
    """

    labels: NDArray[np.int32]
    wm_probability: Optional[NDArray[np.float32]] = None
    gm_probability: Optional[NDArray[np.float32]] = None
    csf_probability: Optional[NDArray[np.float32]] = None
    affine: Optional[NDArray[np.float64]] = None
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def get_tissue_at_point(
        self,
        point: NDArray[np.float64],
    ) -> BrainTissue:
        """Get tissue type at a physical coordinate."""
        voxel = self._physical_to_voxel(point)
        voxel = np.round(voxel).astype(int)

        # Bounds check
        for i in range(3):
            voxel[i] = max(0, min(voxel[i], self.labels.shape[i] - 1))

        label = self.labels[voxel[0], voxel[1], voxel[2]]
        return BrainTissue.from_label(label)

    def get_tissue_probabilities_at_point(
        self,
        point: NDArray[np.float64],
    ) -> Dict[BrainTissue, float]:
        """Get tissue probability distribution at a physical coordinate."""
        voxel = self._physical_to_voxel(point)
        voxel = np.round(voxel).astype(int)

        # Bounds check
        for i in range(3):
            voxel[i] = max(0, min(voxel[i], self.labels.shape[i] - 1))

        probs = {BrainTissue.BACKGROUND: 0.0}

        if self.wm_probability is not None:
            probs[BrainTissue.WHITE_MATTER] = float(
                self.wm_probability[voxel[0], voxel[1], voxel[2]]
            )
        if self.gm_probability is not None:
            probs[BrainTissue.GRAY_MATTER] = float(
                self.gm_probability[voxel[0], voxel[1], voxel[2]]
            )
        if self.csf_probability is not None:
            probs[BrainTissue.CSF] = float(
                self.csf_probability[voxel[0], voxel[1], voxel[2]]
            )

        return probs

    def _physical_to_voxel(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert physical coordinates to voxel coordinates."""
        if self.affine is not None:
            inv_affine = np.linalg.inv(self.affine)
            point_h = np.append(point, 1.0)
            return (inv_affine @ point_h)[:3]
        return point / np.array(self.voxel_size)

    def get_white_matter_mask(self, threshold: float = 0.5) -> NDArray[np.bool_]:
        """Get binary white matter mask."""
        if self.wm_probability is not None:
            return self.wm_probability > threshold
        return self.labels == BrainTissue.WHITE_MATTER.value

    def get_gray_matter_mask(self, threshold: float = 0.5) -> NDArray[np.bool_]:
        """Get binary gray matter mask."""
        if self.gm_probability is not None:
            return self.gm_probability > threshold
        return self.labels == BrainTissue.GRAY_MATTER.value


@dataclass
class AnisotropicMaterialProperties:
    """
    Anisotropic material properties for brain tissue.

    Implements transversely isotropic behavior for white matter,
    where the fiber direction defines the axis of symmetry.

    Attributes:
        E_parallel: Young's modulus along fiber direction (Pa)
        E_perpendicular: Young's modulus perpendicular to fibers (Pa)
        nu_parallel: Poisson's ratio for fiber direction
        nu_perpendicular: Poisson's ratio perpendicular to fibers
        G_parallel: Shear modulus for fiber-aligned shear (Pa)
        fiber_direction: Unit vector of fiber orientation
    """

    # White matter properties (transversely isotropic)
    E_parallel: float = 4000.0  # Pa - stiffer along fibers
    E_perpendicular: float = 2000.0  # Pa - softer perpendicular
    nu_parallel: float = 0.35  # Lower compressibility along fibers
    nu_perpendicular: float = 0.45  # Higher perpendicular (nearly incompressible)
    G_parallel: float = 800.0  # Pa
    fiber_direction: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )

    # Gray matter properties (isotropic, more compressible)
    @classmethod
    def gray_matter_isotropic(cls) -> "AnisotropicMaterialProperties":
        """Create isotropic properties for gray matter."""
        E = 2500.0  # Pa - softer than white matter
        nu = 0.40  # More compressible than near-incompressible
        return cls(
            E_parallel=E,
            E_perpendicular=E,
            nu_parallel=nu,
            nu_perpendicular=nu,
            G_parallel=E / (2 * (1 + nu)),
            fiber_direction=np.array([1.0, 0.0, 0.0]),
        )

    @classmethod
    def white_matter_anisotropic(
        cls,
        fiber_direction: NDArray[np.float64],
        anisotropy_ratio: float = 2.0,
    ) -> "AnisotropicMaterialProperties":
        """
        Create anisotropic properties for white matter.

        Args:
            fiber_direction: Unit vector of local fiber orientation
            anisotropy_ratio: Ratio of parallel to perpendicular stiffness
        """
        E_perp = 2000.0  # Baseline perpendicular stiffness
        E_para = E_perp * anisotropy_ratio  # Stiffer along fibers

        # Normalize fiber direction
        fd = fiber_direction / np.linalg.norm(fiber_direction)

        return cls(
            E_parallel=E_para,
            E_perpendicular=E_perp,
            nu_parallel=0.35,
            nu_perpendicular=0.45,
            G_parallel=E_para / (2 * (1 + 0.35)),
            fiber_direction=fd,
        )

    def get_constitutive_matrix(self) -> NDArray[np.float64]:
        """
        Build the 6x6 constitutive matrix for transversely isotropic material.

        Uses Voigt notation: [σ11, σ22, σ33, σ23, σ13, σ12]

        The material is transversely isotropic with the fiber direction
        as the axis of symmetry (1-direction in local coordinates).
        """
        # Material constants
        E1 = self.E_parallel
        E2 = self.E_perpendicular
        nu12 = self.nu_parallel
        nu23 = self.nu_perpendicular
        G12 = self.G_parallel

        # Derived constants
        # nu21 = nu12 * E2 / E1
        nu21 = nu12 * E2 / E1
        G23 = E2 / (2 * (1 + nu23))

        # Compliance matrix S (strain = S * stress)
        S = np.zeros((6, 6))

        # Normal components
        S[0, 0] = 1 / E1
        S[1, 1] = 1 / E2
        S[2, 2] = 1 / E2

        # Coupling terms
        S[0, 1] = -nu12 / E1
        S[1, 0] = -nu21 / E2
        S[0, 2] = -nu12 / E1
        S[2, 0] = -nu21 / E2
        S[1, 2] = -nu23 / E2
        S[2, 1] = -nu23 / E2

        # Shear components
        S[3, 3] = 1 / G23  # sigma_23
        S[4, 4] = 1 / G12  # sigma_13
        S[5, 5] = 1 / G12  # sigma_12

        # Stiffness matrix C = S^-1
        try:
            C_local = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fall back to isotropic
            E = (E1 + 2 * E2) / 3
            nu = (nu12 + 2 * nu23) / 3
            return self._isotropic_constitutive(E, nu)

        # Rotate to global coordinates using fiber direction
        C_global = self._rotate_constitutive_matrix(C_local, self.fiber_direction)

        return C_global

    def _isotropic_constitutive(self, E: float, nu: float) -> NDArray[np.float64]:
        """Build isotropic constitutive matrix."""
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        C = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ], dtype=np.float64)

        return C

    def _rotate_constitutive_matrix(
        self,
        C_local: NDArray[np.float64],
        fiber_dir: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Rotate constitutive matrix from fiber-aligned to global coordinates.

        The local 1-axis is aligned with the fiber direction.
        """
        # Build rotation matrix from fiber direction
        f = fiber_dir / np.linalg.norm(fiber_dir)

        # Find perpendicular vectors
        if abs(f[0]) < 0.9:
            t = np.array([1, 0, 0])
        else:
            t = np.array([0, 1, 0])

        n1 = np.cross(f, t)
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.cross(f, n1)

        # 3x3 rotation matrix (columns are local axes in global coords)
        R = np.column_stack([f, n1, n2])

        # Build 6x6 transformation matrix for Voigt notation
        T = self._voigt_rotation_matrix(R)

        # Transform: C_global = T^T * C_local * T
        C_global = T.T @ C_local @ T

        return C_global

    def _voigt_rotation_matrix(self, R: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Build 6x6 rotation matrix for Voigt notation stress/strain tensors.

        Transforms [σ11, σ22, σ33, σ23, σ13, σ12] between coordinate systems.
        """
        T = np.zeros((6, 6))

        # Indices for Voigt notation
        # 0: 11, 1: 22, 2: 33, 3: 23, 4: 13, 5: 12
        idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        for I, (i, j) in enumerate(idx):
            for J, (k, l) in enumerate(idx):
                if I < 3 and J < 3:
                    # Normal-normal
                    T[I, J] = R[i, k] * R[j, l]
                elif I < 3 and J >= 3:
                    # Normal-shear
                    T[I, J] = R[i, k] * R[j, l] + R[i, l] * R[j, k]
                elif I >= 3 and J < 3:
                    # Shear-normal
                    T[I, J] = R[i, k] * R[j, l]
                else:
                    # Shear-shear
                    T[I, J] = R[i, k] * R[j, l] + R[i, l] * R[j, k]

        return T


class SUITPyIntegration:
    """
    Integration with SUITPy toolbox for cerebellar atlas access.

    Provides access to SUIT templates, normalization utilities,
    and coordinate transformations between MNI and SUIT spaces.

    Reference: https://github.com/DiedrichsenLab/SUITPy
    """

    SUIT_TEMPLATE_NAME = "tpl-SUIT_T1w.nii.gz"
    SUIT_SHAPE = (91, 109, 91)  # Standard SUIT template dimensions
    SUIT_VOXEL_SIZE = (2.0, 2.0, 2.0)  # mm

    def __init__(self, suit_dir: Optional[Path] = None):
        """
        Initialize SUITPy integration.

        Args:
            suit_dir: Path to SUIT atlas directory. If None, attempts
                     to use SUITPy package to locate default templates.
        """
        self.suit_dir = Path(suit_dir) if suit_dir else None
        self._template_data: Optional[NDArray] = None
        self._template_affine: Optional[NDArray] = None
        self._suitpy_available = self._check_suitpy()

    def _check_suitpy(self) -> bool:
        """Check if SUITPy is installed and functional."""
        try:
            import SUITPy
            return True
        except ImportError:
            return False

    def get_template_path(self) -> Optional[Path]:
        """
        Get path to SUIT T1w template.

        Returns:
            Path to template file or None if not found.
        """
        if self.suit_dir and self.suit_dir.exists():
            template_path = self.suit_dir / self.SUIT_TEMPLATE_NAME
            if template_path.exists():
                return template_path
            # Try alternate naming
            for alt_name in ["SUIT_template.nii.gz", "SUIT_template.nii",
                            "tpl-SUIT_T1w.nii"]:
                alt_path = self.suit_dir / alt_name
                if alt_path.exists():
                    return alt_path

        # Try SUITPy package location
        if self._suitpy_available:
            try:
                import SUITPy
                pkg_dir = Path(SUITPy.__file__).parent
                for subdir in ["data", "atlas", "templates", ""]:
                    if subdir:
                        search_dir = pkg_dir / subdir
                    else:
                        search_dir = pkg_dir
                    if search_dir.exists():
                        for name in [self.SUIT_TEMPLATE_NAME, "SUIT_template.nii.gz"]:
                            template_path = search_dir / name
                            if template_path.exists():
                                return template_path
            except Exception:
                pass

        return None

    def load_template(self) -> Tuple[NDArray[np.float32], NDArray[np.float64]]:
        """
        Load SUIT T1w template.

        Returns:
            Tuple of (template_data, affine_matrix)
        """
        if self._template_data is not None:
            return self._template_data, self._template_affine

        template_path = self.get_template_path()

        if template_path is not None and template_path.exists():
            import nibabel as nib
            img = nib.load(template_path)
            self._template_data = np.asarray(img.get_fdata(), dtype=np.float32)
            self._template_affine = img.affine.astype(np.float64)
        else:
            # Generate synthetic template
            warnings.warn(
                "SUIT template not found. Using synthetic template. "
                "Install SUITPy or provide suit_dir for full functionality.",
                UserWarning
            )
            self._template_data, self._template_affine = self._generate_synthetic_template()

        return self._template_data, self._template_affine

    def _generate_synthetic_template(
        self,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float64]]:
        """Generate synthetic SUIT-like template for testing."""
        shape = self.SUIT_SHAPE
        voxel_size = self.SUIT_VOXEL_SIZE

        template = np.zeros(shape, dtype=np.float32)

        # Create coordinate grids
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        center = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 3])

        # Cerebellar hemispheres
        left_center = center + np.array([-15, 0, 0])
        right_center = center + np.array([15, 0, 0])

        for hemi_center, intensity in [(left_center, 100), (right_center, 100)]:
            dist = ((x - hemi_center[0]) / 20) ** 2 + \
                   ((y - hemi_center[1]) / 25) ** 2 + \
                   ((z - hemi_center[2]) / 15) ** 2
            template[dist < 1] = intensity

        # Vermis
        vermis_dist = ((x - center[0]) / 8) ** 2 + \
                      ((y - center[1]) / 20) ** 2 + \
                      ((z - center[2]) / 12) ** 2
        template[vermis_dist < 1] = 110

        # Add noise
        noise = np.random.normal(0, 3, shape).astype(np.float32)
        template = np.clip(template + noise * (template > 0), 0, 255)

        # Create affine
        affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
        affine[:3, 3] = -np.array(shape) * np.array(voxel_size) / 2

        return template, affine.astype(np.float64)

    def get_suit_affine(self) -> NDArray[np.float64]:
        """Get SUIT template affine transformation matrix."""
        _, affine = self.load_template()
        return affine


class MNIAtlasLoader:
    """
    Loader for MNI-space atlases including tissue segmentation and DTI.

    Supports:
    - ICBM152 tissue probability maps (GM, WM, CSF)
    - HCP1065 DTI atlas for fiber orientation
    - FSL atlas integration

    References:
    - ICBM152: http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
    - HCP1065 DTI: Warrington et al., FSL (derived from Human Connectome Project)
    """

    # Standard MNI152 template dimensions
    MNI_SHAPE = (182, 218, 182)
    MNI_VOXEL_SIZE = (1.0, 1.0, 1.0)

    # FSL atlas paths (relative to FSL data directory)
    FSL_TISSUE_PRIOR_PATH = "standard/tissuepriors"
    FSL_JHU_ATLAS_PATH = "atlases/JHU"

    # Default paths to bundled atlas files
    DEFAULT_MNI_DIR = Path(__file__).parent.parent.parent / "data" / "atlases" / "MNI152"
    DEFAULT_DTI_DIR = Path(__file__).parent.parent.parent / "data" / "atlases" / "HCP1065_DTI"

    def __init__(
        self,
        fsl_dir: Optional[Path] = None,
        mni_template_dir: Optional[Path] = None,
        dti_template_dir: Optional[Path] = None,
        use_bundled: bool = True,
    ):
        """
        Initialize MNI atlas loader.

        Args:
            fsl_dir: Path to FSL installation directory (e.g., /usr/local/fsl).
                    If None, attempts to detect from FSLDIR environment variable.
            mni_template_dir: Alternative path to MNI tissue templates.
            dti_template_dir: Alternative path to DTI templates.
            use_bundled: If True and templates not found elsewhere, use bundled
                        atlas files from the data/atlases directory.
        """
        self.fsl_dir = self._find_fsl_dir(fsl_dir)
        self.mni_template_dir = Path(mni_template_dir) if mni_template_dir else None
        self.dti_template_dir = Path(dti_template_dir) if dti_template_dir else None
        self.use_bundled = use_bundled
        self._cached_segmentation: Optional[TissueSegmentation] = None
        self._cached_fibers: Optional[FiberOrientation] = None

    def _find_fsl_dir(self, fsl_dir: Optional[Path]) -> Optional[Path]:
        """Find FSL installation directory."""
        if fsl_dir:
            return Path(fsl_dir)

        import os
        fsl_env = os.environ.get("FSLDIR")
        if fsl_env:
            return Path(fsl_env)

        # Common installation paths
        for path in ["/usr/local/fsl", "/opt/fsl", "/usr/share/fsl"]:
            if Path(path).exists():
                return Path(path)

        return None

    def load_tissue_segmentation(self) -> TissueSegmentation:
        """
        Load tissue segmentation (GM, WM, CSF probability maps).

        Returns:
            TissueSegmentation with probability maps and labels.
        """
        if self._cached_segmentation is not None:
            return self._cached_segmentation

        # Try to load from FSL
        segmentation = self._load_fsl_tissue_priors()

        if segmentation is None:
            # Try alternative location
            segmentation = self._load_alternative_segmentation()

        if segmentation is None:
            # Generate synthetic segmentation
            warnings.warn(
                "Tissue probability maps not found. Using synthetic segmentation. "
                "Install FSL or provide tissue priors for accurate modeling.",
                UserWarning
            )
            segmentation = self._generate_synthetic_segmentation()

        self._cached_segmentation = segmentation
        return segmentation

    def _load_fsl_tissue_priors(self) -> Optional[TissueSegmentation]:
        """Load tissue priors from FSL installation."""
        if self.fsl_dir is None:
            return None

        try:
            import nibabel as nib
        except ImportError:
            return None

        data_dir = self.fsl_dir / "data"

        # Try standard tissue prior locations
        prior_dirs = [
            data_dir / "standard" / "tissuepriors",
            data_dir / "atlases" / "MNI",
            data_dir / "standard",
        ]

        gm_data, wm_data, csf_data = None, None, None
        affine = None

        for prior_dir in prior_dirs:
            if not prior_dir.exists():
                continue

            # Common naming patterns
            gm_names = ["avg152T1_gray.nii.gz", "gray.nii.gz",
                       "MNI152_T1_1mm_brain_pve_1.nii.gz"]
            wm_names = ["avg152T1_white.nii.gz", "white.nii.gz",
                       "MNI152_T1_1mm_brain_pve_2.nii.gz"]
            csf_names = ["avg152T1_csf.nii.gz", "csf.nii.gz",
                        "MNI152_T1_1mm_brain_pve_0.nii.gz"]

            for name in gm_names:
                gm_path = prior_dir / name
                if gm_path.exists():
                    img = nib.load(gm_path)
                    gm_data = np.asarray(img.get_fdata(), dtype=np.float32)
                    affine = img.affine
                    break

            for name in wm_names:
                wm_path = prior_dir / name
                if wm_path.exists():
                    img = nib.load(wm_path)
                    wm_data = np.asarray(img.get_fdata(), dtype=np.float32)
                    if affine is None:
                        affine = img.affine
                    break

            for name in csf_names:
                csf_path = prior_dir / name
                if csf_path.exists():
                    img = nib.load(csf_path)
                    csf_data = np.asarray(img.get_fdata(), dtype=np.float32)
                    if affine is None:
                        affine = img.affine
                    break

            if gm_data is not None or wm_data is not None:
                break

        if gm_data is None and wm_data is None:
            return None

        # Create discrete labels from probabilities
        shape = (gm_data if gm_data is not None else wm_data).shape
        labels = np.zeros(shape, dtype=np.int32)

        # Assign labels based on highest probability
        if csf_data is not None:
            labels[csf_data > 0.5] = BrainTissue.CSF.value
        if gm_data is not None:
            labels[gm_data > 0.5] = BrainTissue.GRAY_MATTER.value
        if wm_data is not None:
            labels[wm_data > 0.5] = BrainTissue.WHITE_MATTER.value

        # Determine voxel size from affine
        voxel_size = tuple(np.abs(np.diag(affine)[:3]).tolist()) if affine is not None else self.MNI_VOXEL_SIZE

        return TissueSegmentation(
            labels=labels,
            wm_probability=wm_data,
            gm_probability=gm_data,
            csf_probability=csf_data,
            affine=affine,
            voxel_size=voxel_size,
        )

    def _load_alternative_segmentation(self) -> Optional[TissueSegmentation]:
        """Load segmentation from alternative or bundled locations."""
        try:
            import nibabel as nib
        except ImportError:
            return None

        # Try user-specified directory first
        search_dirs = []
        if self.mni_template_dir is not None:
            search_dirs.append(Path(self.mni_template_dir))

        # Try bundled files
        if self.use_bundled and self.DEFAULT_MNI_DIR.exists():
            search_dirs.append(self.DEFAULT_MNI_DIR)

        for template_dir in search_dirs:
            if not template_dir.exists():
                continue

            # Try to load bundled MNI152 files (from Jfortin1/MNITemplate)
            seg_path = template_dir / "MNI152_T1_1mm_Brain_FAST_seg.nii.gz"
            pve0_path = template_dir / "MNI152_T1_1mm_Brain_FAST_pve_0.nii.gz"  # CSF
            pve1_path = template_dir / "MNI152_T1_1mm_Brain_FAST_pve_1.nii.gz"  # GM
            pve2_path = template_dir / "MNI152_T1_1mm_Brain_FAST_pve_2.nii.gz"  # WM

            if seg_path.exists():
                seg_img = nib.load(seg_path)
                labels = np.asarray(seg_img.get_fdata(), dtype=np.int32)
                affine = seg_img.affine
                voxel_size = tuple(np.abs(np.diag(affine)[:3]).tolist())

                csf_prob = None
                gm_prob = None
                wm_prob = None

                if pve0_path.exists():
                    csf_prob = np.asarray(nib.load(pve0_path).get_fdata(), dtype=np.float32)
                if pve1_path.exists():
                    gm_prob = np.asarray(nib.load(pve1_path).get_fdata(), dtype=np.float32)
                if pve2_path.exists():
                    wm_prob = np.asarray(nib.load(pve2_path).get_fdata(), dtype=np.float32)

                return TissueSegmentation(
                    labels=labels,
                    wm_probability=wm_prob,
                    gm_probability=gm_prob,
                    csf_probability=csf_prob,
                    affine=affine,
                    voxel_size=voxel_size,
                )

        return None

    def _generate_synthetic_segmentation(self) -> TissueSegmentation:
        """Generate synthetic tissue segmentation for testing."""
        shape = self.MNI_SHAPE

        # Create coordinate grids
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        center = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2])

        # Create brain mask (ellipsoid)
        brain_dist = ((x - center[0]) / 70) ** 2 + \
                     ((y - center[1]) / 85) ** 2 + \
                     ((z - center[2]) / 60) ** 2
        brain_mask = brain_dist < 1

        # Gray matter: outer shell of brain
        gm_dist = ((x - center[0]) / 65) ** 2 + \
                  ((y - center[1]) / 80) ** 2 + \
                  ((z - center[2]) / 55) ** 2
        gm_mask = brain_mask & (gm_dist > 0.7)

        # White matter: inner core
        wm_mask = brain_mask & (gm_dist <= 0.7)

        # CSF: ventricles (simple ellipsoid in center)
        csf_dist = ((x - center[0]) / 10) ** 2 + \
                   ((y - center[1] - 10) / 15) ** 2 + \
                   ((z - center[2]) / 8) ** 2
        csf_mask = csf_dist < 1

        # Update masks
        wm_mask = wm_mask & ~csf_mask

        # Create label volume
        labels = np.zeros(shape, dtype=np.int32)
        labels[csf_mask] = BrainTissue.CSF.value
        labels[gm_mask] = BrainTissue.GRAY_MATTER.value
        labels[wm_mask] = BrainTissue.WHITE_MATTER.value

        # Create probability maps
        gm_prob = np.zeros(shape, dtype=np.float32)
        wm_prob = np.zeros(shape, dtype=np.float32)
        csf_prob = np.zeros(shape, dtype=np.float32)

        gm_prob[gm_mask] = 1.0
        wm_prob[wm_mask] = 1.0
        csf_prob[csf_mask] = 1.0

        # Add gradients at boundaries
        from scipy import ndimage
        gm_prob = ndimage.gaussian_filter(gm_prob.astype(float), sigma=2).astype(np.float32)
        wm_prob = ndimage.gaussian_filter(wm_prob.astype(float), sigma=2).astype(np.float32)
        csf_prob = ndimage.gaussian_filter(csf_prob.astype(float), sigma=2).astype(np.float32)

        # Normalize
        total = gm_prob + wm_prob + csf_prob + 1e-6
        gm_prob = gm_prob / total
        wm_prob = wm_prob / total
        csf_prob = csf_prob / total

        affine = np.eye(4)
        affine[:3, 3] = -np.array(shape) / 2

        return TissueSegmentation(
            labels=labels,
            wm_probability=wm_prob,
            gm_probability=gm_prob,
            csf_probability=csf_prob,
            affine=affine,
            voxel_size=self.MNI_VOXEL_SIZE,
        )

    def load_fiber_orientation(self) -> FiberOrientation:
        """
        Load fiber orientation data from JHU DTI atlas.

        Returns:
            FiberOrientation with principal diffusion directions.
        """
        if self._cached_fibers is not None:
            return self._cached_fibers

        # Try to load from FSL
        fibers = self._load_jhu_dti_atlas()

        if fibers is None:
            # Generate synthetic fiber field
            warnings.warn(
                "JHU DTI atlas not found. Using synthetic fiber orientations. "
                "Install FSL or provide DTI data for accurate white matter modeling.",
                UserWarning
            )
            fibers = self._generate_synthetic_fibers()

        self._cached_fibers = fibers
        return fibers

    def _load_jhu_dti_atlas(self) -> Optional[FiberOrientation]:
        """Load DTI atlas from FSL or bundled HCP1065 files."""
        try:
            import nibabel as nib
        except ImportError:
            return None

        # Build list of directories to search
        search_dirs = []

        # User-specified DTI directory
        if self.dti_template_dir is not None:
            search_dirs.append(Path(self.dti_template_dir))

        # FSL JHU atlas
        if self.fsl_dir is not None:
            search_dirs.append(self.fsl_dir / "data" / "atlases" / "JHU")

        # Bundled HCP1065 DTI atlas
        if self.use_bundled and self.DEFAULT_DTI_DIR.exists():
            search_dirs.append(self.DEFAULT_DTI_DIR)

        for dti_dir in search_dirs:
            if not dti_dir.exists():
                continue

            # Try HCP1065 naming convention (bundled files)
            fa_path = dti_dir / "FSL_HCP1065_FA_1mm.nii.gz"
            v1_path = dti_dir / "FSL_HCP1065_V1_1mm.nii.gz"

            # Fall back to JHU naming convention
            if not fa_path.exists():
                fa_path = dti_dir / "JHU-ICBM-FA-1mm.nii.gz"
            if not fa_path.exists():
                fa_path = dti_dir / "JHU-ICBM-FA-2mm.nii.gz"
            if not v1_path.exists():
                v1_path = dti_dir / "JHU-ICBM-V1-1mm.nii.gz"

            fa_data = None
            affine = None
            vectors = None

            if fa_path.exists():
                fa_img = nib.load(fa_path)
                fa_data = np.asarray(fa_img.get_fdata(), dtype=np.float32)
                affine = fa_img.affine

            if v1_path.exists():
                v1_img = nib.load(v1_path)
                vectors = np.asarray(v1_img.get_fdata(), dtype=np.float64)
                if affine is None:
                    affine = v1_img.affine

            if vectors is None and fa_data is not None:
                # Generate synthetic directions based on FA
                vectors = self._estimate_fibers_from_fa(fa_data)

            if vectors is not None:
                return FiberOrientation(
                    vectors=vectors,
                    fractional_anisotropy=fa_data if fa_data is not None else np.ones(vectors.shape[:3], dtype=np.float32) * 0.5,
                    affine=affine,
                )

        return None

    def _estimate_fibers_from_fa(
        self,
        fa_data: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Estimate fiber directions from FA using gradient."""
        from scipy import ndimage

        shape = fa_data.shape
        vectors = np.zeros((*shape, 3), dtype=np.float64)

        # Use image gradient direction as proxy for perpendicular-to-fiber
        # (actual fibers require full tensor)
        grad_x = ndimage.sobel(fa_data, axis=0)
        grad_y = ndimage.sobel(fa_data, axis=1)
        grad_z = ndimage.sobel(fa_data, axis=2)

        # Stack gradients
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2) + 1e-6

        # Fiber direction perpendicular to gradient (approximation)
        # For corpus callosum: left-right
        # For corticospinal: superior-inferior
        vectors[..., 0] = -grad_y / grad_mag  # Perpendicular in x
        vectors[..., 1] = grad_x / grad_mag   # Perpendicular in y
        vectors[..., 2] = 0.1  # Slight z component

        # Normalize
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms < 1e-6] = 1
        vectors = vectors / norms

        # Use default direction where FA is low
        low_fa_mask = fa_data < 0.2
        vectors[low_fa_mask] = [1, 0, 0]

        return vectors

    def _generate_synthetic_fibers(self) -> FiberOrientation:
        """Generate synthetic fiber orientation field for testing."""
        shape = self.MNI_SHAPE

        vectors = np.zeros((*shape, 3), dtype=np.float64)
        fa = np.zeros(shape, dtype=np.float32)

        center = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2])

        # Create coordinate grids
        X, Y, Z = np.meshgrid(
            np.arange(shape[0]) - center[0],
            np.arange(shape[1]) - center[1],
            np.arange(shape[2]) - center[2],
            indexing='ij'
        )

        # Corpus callosum region: left-right fibers
        cc_mask = (np.abs(Z) < 10) & (np.abs(Y + 10) < 15) & (np.abs(X) < 40)
        vectors[cc_mask, 0] = 1.0  # Left-right
        fa[cc_mask] = 0.8

        # Corticospinal tract: superior-inferior
        cst_left = ((X + 25)**2 + Y**2 < 100) & (Z > -30) & (Z < 50)
        cst_right = ((X - 25)**2 + Y**2 < 100) & (Z > -30) & (Z < 50)
        vectors[cst_left | cst_right, 2] = 1.0  # Superior-inferior
        fa[cst_left | cst_right] = 0.7

        # Default: anterior-posterior with some variation
        default_mask = (fa < 0.1) & (np.sqrt(X**2 + Y**2 + Z**2) < 70)
        vectors[default_mask, 1] = 0.8
        vectors[default_mask, 2] = 0.2
        fa[default_mask] = 0.3

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms < 1e-6] = 1
        vectors = vectors / norms

        affine = np.eye(4)
        affine[:3, 3] = -np.array(shape) / 2

        return FiberOrientation(
            vectors=vectors,
            fractional_anisotropy=fa,
            affine=affine,
        )


class SpaceTransformer:
    """
    Coordinate transformation between MNI and SUIT spaces.

    Handles resampling of tissue segmentation and fiber orientation
    from MNI152 space to SUIT cerebellar space.
    """

    def __init__(
        self,
        suit_integration: SUITPyIntegration,
        mni_loader: MNIAtlasLoader,
    ):
        """
        Initialize space transformer.

        Args:
            suit_integration: SUITPy integration instance
            mni_loader: MNI atlas loader instance
        """
        self.suit = suit_integration
        self.mni = mni_loader
        self._transform_matrix: Optional[NDArray[np.float64]] = None

    def mni_to_suit_coords(
        self,
        mni_coords: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Transform coordinates from MNI to SUIT space.

        Args:
            mni_coords: Coordinates in MNI space, shape (N, 3) or (3,)

        Returns:
            Coordinates in SUIT space
        """
        if self._transform_matrix is None:
            self._transform_matrix = self._compute_transform()

        single = mni_coords.ndim == 1
        if single:
            mni_coords = mni_coords.reshape(1, 3)

        # Apply affine transformation
        ones = np.ones((len(mni_coords), 1))
        coords_h = np.hstack([mni_coords, ones])
        suit_coords_h = (self._transform_matrix @ coords_h.T).T
        suit_coords = suit_coords_h[:, :3]

        if single:
            return suit_coords[0]
        return suit_coords

    def suit_to_mni_coords(
        self,
        suit_coords: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Transform coordinates from SUIT to MNI space.

        Args:
            suit_coords: Coordinates in SUIT space, shape (N, 3) or (3,)

        Returns:
            Coordinates in MNI space
        """
        if self._transform_matrix is None:
            self._transform_matrix = self._compute_transform()

        single = suit_coords.ndim == 1
        if single:
            suit_coords = suit_coords.reshape(1, 3)

        # Apply inverse transformation
        inv_transform = np.linalg.inv(self._transform_matrix)
        ones = np.ones((len(suit_coords), 1))
        coords_h = np.hstack([suit_coords, ones])
        mni_coords_h = (inv_transform @ coords_h.T).T
        mni_coords = mni_coords_h[:, :3]

        if single:
            return mni_coords[0]
        return mni_coords

    def _compute_transform(self) -> NDArray[np.float64]:
        """
        Compute transformation matrix from MNI to SUIT space.

        The SUIT template is in a slightly different space than MNI152,
        with the origin at the anterior commissure and focused on the
        posterior fossa.
        """
        # Get SUIT affine
        suit_affine = self.suit.get_suit_affine()

        # Standard MNI to SUIT transformation
        # This is an approximation - for accurate transformation,
        # use the nonlinear warp from SUITPy

        # SUIT space is focused on cerebellum, roughly:
        # - Origin shifted inferiorly and posteriorly
        # - Same orientation as MNI

        # Simple affine approximation (linear component of full warp)
        # Translation to focus on posterior fossa
        mni_to_suit = np.eye(4)
        mni_to_suit[:3, 3] = [0, 20, -40]  # Shift focus to cerebellum

        # Combine with SUIT affine
        transform = suit_affine @ mni_to_suit

        return transform

    def resample_segmentation_to_suit(
        self,
        segmentation: TissueSegmentation,
    ) -> TissueSegmentation:
        """
        Resample tissue segmentation from MNI to SUIT space.

        Args:
            segmentation: Tissue segmentation in MNI space

        Returns:
            Tissue segmentation resampled to SUIT space
        """
        from scipy import ndimage

        suit_shape = self.suit.SUIT_SHAPE
        suit_voxel_size = self.suit.SUIT_VOXEL_SIZE
        suit_affine = self.suit.get_suit_affine()

        # Build coordinate grid in SUIT space
        coords = np.zeros((*suit_shape, 3))
        for i in range(suit_shape[0]):
            for j in range(suit_shape[1]):
                for k in range(suit_shape[2]):
                    # SUIT voxel to physical
                    suit_phys = suit_affine @ np.array([i, j, k, 1])
                    # Physical to MNI voxel
                    mni_phys = self.suit_to_mni_coords(suit_phys[:3])
                    if segmentation.affine is not None:
                        mni_vox = np.linalg.inv(segmentation.affine) @ np.append(mni_phys, 1)
                        coords[i, j, k] = mni_vox[:3]
                    else:
                        coords[i, j, k] = mni_phys / np.array(segmentation.voxel_size)

        # Resample using map_coordinates
        # Labels: nearest neighbor
        new_labels = ndimage.map_coordinates(
            segmentation.labels.astype(float),
            [coords[..., 0], coords[..., 1], coords[..., 2]],
            order=0,  # Nearest neighbor
            mode='constant',
            cval=0,
        ).astype(np.int32)

        # Probability maps: linear interpolation
        new_wm_prob = None
        new_gm_prob = None
        new_csf_prob = None

        if segmentation.wm_probability is not None:
            new_wm_prob = ndimage.map_coordinates(
                segmentation.wm_probability,
                [coords[..., 0], coords[..., 1], coords[..., 2]],
                order=1,
                mode='constant',
                cval=0,
            ).astype(np.float32)

        if segmentation.gm_probability is not None:
            new_gm_prob = ndimage.map_coordinates(
                segmentation.gm_probability,
                [coords[..., 0], coords[..., 1], coords[..., 2]],
                order=1,
                mode='constant',
                cval=0,
            ).astype(np.float32)

        if segmentation.csf_probability is not None:
            new_csf_prob = ndimage.map_coordinates(
                segmentation.csf_probability,
                [coords[..., 0], coords[..., 1], coords[..., 2]],
                order=1,
                mode='constant',
                cval=0,
            ).astype(np.float32)

        return TissueSegmentation(
            labels=new_labels,
            wm_probability=new_wm_prob,
            gm_probability=new_gm_prob,
            csf_probability=new_csf_prob,
            affine=suit_affine,
            voxel_size=suit_voxel_size,
        )

    def resample_fibers_to_suit(
        self,
        fibers: FiberOrientation,
    ) -> FiberOrientation:
        """
        Resample fiber orientation from MNI to SUIT space.

        Args:
            fibers: Fiber orientation in MNI space

        Returns:
            Fiber orientation resampled to SUIT space
        """
        from scipy import ndimage

        suit_shape = self.suit.SUIT_SHAPE
        suit_affine = self.suit.get_suit_affine()

        # Build coordinate grid
        coords = np.zeros((*suit_shape, 3))
        for i in range(suit_shape[0]):
            for j in range(suit_shape[1]):
                for k in range(suit_shape[2]):
                    suit_phys = suit_affine @ np.array([i, j, k, 1])
                    mni_phys = self.suit_to_mni_coords(suit_phys[:3])
                    if fibers.affine is not None:
                        mni_vox = np.linalg.inv(fibers.affine) @ np.append(mni_phys, 1)
                        coords[i, j, k] = mni_vox[:3]
                    else:
                        coords[i, j, k] = mni_phys

        # Resample vector components
        if fibers.vectors.ndim == 4:
            new_vectors = np.zeros((*suit_shape, 3), dtype=np.float64)
            for d in range(3):
                new_vectors[..., d] = ndimage.map_coordinates(
                    fibers.vectors[..., d],
                    [coords[..., 0], coords[..., 1], coords[..., 2]],
                    order=1,
                    mode='constant',
                    cval=0,
                )

            # Renormalize
            norms = np.linalg.norm(new_vectors, axis=-1, keepdims=True)
            norms[norms < 1e-6] = 1
            new_vectors = new_vectors / norms
        else:
            # 2D array of vectors
            new_vectors = fibers.vectors  # Keep as is for node-based

        # Resample FA
        new_fa = ndimage.map_coordinates(
            fibers.fractional_anisotropy,
            [coords[..., 0], coords[..., 1], coords[..., 2]],
            order=1,
            mode='constant',
            cval=0,
        ).astype(np.float32)

        return FiberOrientation(
            vectors=new_vectors,
            fractional_anisotropy=new_fa,
            affine=suit_affine,
        )


class BiophysicalConstraints:
    """
    Main class for managing biophysical constraints in brain FEM modeling.

    Integrates tissue segmentation, fiber orientation, and boundary conditions
    to provide physically realistic material properties for the FEM solver.

    Default configuration uses ICBM-152 MNI space with modeling restricted
    to the posterior fossa (cerebellum + brainstem).

    Attributes:
        tumor_origin: Default tumor seed location in MNI coordinates
        posterior_fossa_only: If True, only model brainstem and cerebellum
    """

    # Default tumor origin: vermis in MNI space
    DEFAULT_TUMOR_ORIGIN = DEFAULT_TUMOR_ORIGIN_MNI

    def __init__(
        self,
        suit_dir: Optional[Path] = None,
        fsl_dir: Optional[Path] = None,
        use_suit_space: bool = False,  # Default to MNI/ICBM152 space
        posterior_fossa_only: bool = True,  # Only model cerebellum + brainstem
        tumor_origin: Optional[NDArray[np.float64]] = None,
    ):
        """
        Initialize biophysical constraints.

        Args:
            suit_dir: Path to SUIT atlas directory (optional)
            fsl_dir: Path to FSL installation (optional, uses FSLDIR env var)
            use_suit_space: If True, resample MNI data to SUIT space.
                           Default is False (use ICBM-152 MNI space).
            posterior_fossa_only: If True, only include cerebellum and brainstem
                                 tissues in the model. Default is True.
            tumor_origin: Tumor seed location in MNI coordinates.
                         Default is [1, -61, -34] (vermis).
        """
        self.suit = SUITPyIntegration(suit_dir)
        self.mni = MNIAtlasLoader(fsl_dir)
        self.transformer = SpaceTransformer(self.suit, self.mni)
        self.use_suit_space = use_suit_space
        self.posterior_fossa_only = posterior_fossa_only

        # Set tumor origin (default: vermis in MNI space)
        self.tumor_origin = tumor_origin if tumor_origin is not None else self.DEFAULT_TUMOR_ORIGIN.copy()

        self._segmentation: Optional[TissueSegmentation] = None
        self._fibers: Optional[FiberOrientation] = None
        self._skull_boundary: Optional[NDArray[np.bool_]] = None
        self._posterior_fossa_mask: Optional[NDArray[np.bool_]] = None

    def load_all_constraints(self) -> None:
        """Load all biophysical constraint data."""
        if self.posterior_fossa_only:
            self.compute_posterior_fossa_mask()
        self.load_tissue_segmentation()
        self.load_fiber_orientation()
        self.compute_skull_boundary()

    def compute_posterior_fossa_mask(self) -> NDArray[np.bool_]:
        """
        Compute mask for posterior fossa region (cerebellum + brainstem).

        The posterior fossa is defined by anatomical bounds in MNI space,
        covering the cerebellum and brainstem while excluding supratentorial
        structures.

        Returns:
            Boolean mask indicating posterior fossa voxels
        """
        if self._posterior_fossa_mask is not None:
            return self._posterior_fossa_mask

        # Get the shape and affine from MNI loader
        shape = self.mni.MNI_SHAPE
        voxel_size = self.mni.MNI_VOXEL_SIZE

        # Create coordinate grids in MNI physical space
        # Standard MNI affine: origin at AC, 1mm isotropic
        # Voxel [91, 109, 91] corresponds to [0, 0, 0] in MNI space for 182x218x182 template
        center_voxel = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2])

        mask = np.zeros(shape, dtype=np.bool_)

        bounds = POSTERIOR_FOSSA_BOUNDS_MNI

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    # Convert voxel to MNI coordinates
                    mni_x = (i - center_voxel[0]) * voxel_size[0]
                    mni_y = (j - center_voxel[1]) * voxel_size[1]
                    mni_z = (k - center_voxel[2]) * voxel_size[2]

                    # Check if within posterior fossa bounds
                    in_bounds = (
                        bounds['x_min'] <= mni_x <= bounds['x_max'] and
                        bounds['y_min'] <= mni_y <= bounds['y_max'] and
                        bounds['z_min'] <= mni_z <= bounds['z_max']
                    )

                    if in_bounds:
                        mask[i, j, k] = True

        self._posterior_fossa_mask = mask
        return self._posterior_fossa_mask

    def get_posterior_fossa_bounds(self) -> Dict[str, float]:
        """Get the MNI coordinate bounds for the posterior fossa region."""
        return POSTERIOR_FOSSA_BOUNDS_MNI.copy()

    def is_in_posterior_fossa(self, mni_coords: NDArray[np.float64]) -> bool:
        """
        Check if MNI coordinates are within the posterior fossa.

        Args:
            mni_coords: Coordinates in MNI space [x, y, z]

        Returns:
            True if coordinates are within posterior fossa bounds
        """
        bounds = POSTERIOR_FOSSA_BOUNDS_MNI
        return (
            bounds['x_min'] <= mni_coords[0] <= bounds['x_max'] and
            bounds['y_min'] <= mni_coords[1] <= bounds['y_max'] and
            bounds['z_min'] <= mni_coords[2] <= bounds['z_max']
        )

    def load_tissue_segmentation(self) -> TissueSegmentation:
        """
        Load tissue segmentation, optionally restricted to posterior fossa.

        If posterior_fossa_only is True, tissues outside the cerebellum and
        brainstem region are masked out (set to background).
        """
        if self._segmentation is not None:
            return self._segmentation

        mni_seg = self.mni.load_tissue_segmentation()

        # Apply posterior fossa mask if requested
        if self.posterior_fossa_only:
            mni_seg = self._apply_posterior_fossa_mask(mni_seg)

        if self.use_suit_space:
            self._segmentation = self.transformer.resample_segmentation_to_suit(mni_seg)
        else:
            self._segmentation = mni_seg

        return self._segmentation

    def _apply_posterior_fossa_mask(
        self,
        segmentation: TissueSegmentation,
    ) -> TissueSegmentation:
        """
        Apply posterior fossa mask to tissue segmentation.

        Sets all tissues outside the posterior fossa to background.

        Args:
            segmentation: Full brain tissue segmentation

        Returns:
            Tissue segmentation restricted to posterior fossa
        """
        pf_mask = self.compute_posterior_fossa_mask()

        # Ensure mask matches segmentation shape
        if pf_mask.shape != segmentation.labels.shape:
            from scipy import ndimage
            # Resample mask to match segmentation
            zoom_factors = np.array(segmentation.labels.shape) / np.array(pf_mask.shape)
            pf_mask = ndimage.zoom(pf_mask.astype(float), zoom_factors, order=0) > 0.5

        # Apply mask to labels
        masked_labels = segmentation.labels.copy()
        masked_labels[~pf_mask] = BrainTissue.BACKGROUND.value

        # Apply mask to probability maps
        masked_wm = None
        masked_gm = None
        masked_csf = None

        if segmentation.wm_probability is not None:
            masked_wm = segmentation.wm_probability.copy()
            masked_wm[~pf_mask] = 0.0

        if segmentation.gm_probability is not None:
            masked_gm = segmentation.gm_probability.copy()
            masked_gm[~pf_mask] = 0.0

        if segmentation.csf_probability is not None:
            masked_csf = segmentation.csf_probability.copy()
            masked_csf[~pf_mask] = 0.0

        return TissueSegmentation(
            labels=masked_labels,
            wm_probability=masked_wm,
            gm_probability=masked_gm,
            csf_probability=masked_csf,
            affine=segmentation.affine,
            voxel_size=segmentation.voxel_size,
        )

    def load_fiber_orientation(self) -> FiberOrientation:
        """Load and optionally resample fiber orientation."""
        if self._fibers is not None:
            return self._fibers

        mni_fibers = self.mni.load_fiber_orientation()

        if self.use_suit_space:
            self._fibers = self.transformer.resample_fibers_to_suit(mni_fibers)
        else:
            self._fibers = mni_fibers

        return self._fibers

    def compute_skull_boundary(self) -> NDArray[np.bool_]:
        """
        Compute skull boundary mask for immovable boundary condition.

        Returns:
            Boolean mask indicating skull voxels
        """
        if self._skull_boundary is not None:
            return self._skull_boundary

        segmentation = self.load_tissue_segmentation()

        # Skull is the outer boundary of the brain
        # Dilate the brain mask and take the shell
        from scipy import ndimage

        brain_mask = segmentation.labels > 0
        dilated = ndimage.binary_dilation(brain_mask, iterations=3)
        self._skull_boundary = dilated & ~brain_mask

        return self._skull_boundary

    def get_material_properties_at_node(
        self,
        position: NDArray[np.float64],
    ) -> AnisotropicMaterialProperties:
        """
        Get material properties for a node based on its position.

        Args:
            position: Physical coordinates of the node (mm)

        Returns:
            AnisotropicMaterialProperties for this location
        """
        segmentation = self.load_tissue_segmentation()
        fibers = self.load_fiber_orientation()

        tissue = segmentation.get_tissue_at_point(position)

        if tissue == BrainTissue.WHITE_MATTER:
            # Get local fiber direction
            fiber_dir = fibers.get_direction_at_point(
                position, segmentation.voxel_size
            )
            return AnisotropicMaterialProperties.white_matter_anisotropic(
                fiber_direction=fiber_dir,
                anisotropy_ratio=2.0,
            )
        elif tissue == BrainTissue.GRAY_MATTER:
            return AnisotropicMaterialProperties.gray_matter_isotropic()
        elif tissue == BrainTissue.CSF:
            # CSF is nearly incompressible fluid
            return AnisotropicMaterialProperties(
                E_parallel=100.0,  # Very soft
                E_perpendicular=100.0,
                nu_parallel=0.49,  # Nearly incompressible
                nu_perpendicular=0.49,
                G_parallel=100.0 / 3,
            )
        else:
            # Default: soft tissue
            return AnisotropicMaterialProperties.gray_matter_isotropic()

    def is_skull_boundary(self, position: NDArray[np.float64]) -> bool:
        """Check if a position is on the skull boundary."""
        skull = self.compute_skull_boundary()
        segmentation = self.load_tissue_segmentation()

        voxel = segmentation._physical_to_voxel(position)
        voxel = np.round(voxel).astype(int)

        for i in range(3):
            voxel[i] = max(0, min(voxel[i], skull.shape[i] - 1))

        return skull[voxel[0], voxel[1], voxel[2]]

    def _compute_anterior_brainstem_boundary(
        self,
        mesh_nodes: NDArray[np.float64],
    ) -> NDArray[np.int32]:
        """
        Identify nodes on the anterior boundary of the brainstem.

        The anterior surface of the brainstem is bounded by the clivus (petrous
        bone), which should act as a hard constraint similar to the skull. This
        prevents unrealistic anterior tumor expansion.

        In MNI coordinates:
        - Y axis: anterior-posterior (positive Y = anterior)
        - The brainstem's anterior surface faces the clivus

        Approach:
        - For each X-Z column in the mesh, find the node with maximum Y
        - These nodes form the anterior boundary

        Args:
            mesh_nodes: Node positions in MNI coordinates, shape (N, 3)

        Returns:
            Indices of nodes on the anterior brainstem boundary
        """
        if len(mesh_nodes) == 0:
            return np.array([], dtype=np.int32)

        # Use coarse binning to find anterior-most nodes in each X-Z column
        # Bin size of 3mm matches typical coarse mesh resolution
        bin_size = 3.0

        # Get X-Z range
        x_coords = mesh_nodes[:, 0]
        z_coords = mesh_nodes[:, 2]
        y_coords = mesh_nodes[:, 1]

        # Create bins for X-Z grid
        x_min, x_max = x_coords.min(), x_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()

        # Dictionary to store max-Y node index for each X-Z bin
        anterior_nodes: Dict[Tuple[int, int], Tuple[int, float]] = {}

        for node_idx, pos in enumerate(mesh_nodes):
            # Compute bin indices
            x_bin = int((pos[0] - x_min) / bin_size)
            z_bin = int((pos[2] - z_min) / bin_size)
            bin_key = (x_bin, z_bin)

            y_val = pos[1]  # Y coordinate (anterior-posterior)

            # Keep track of node with maximum Y (most anterior) in each bin
            if bin_key not in anterior_nodes or y_val > anterior_nodes[bin_key][1]:
                anterior_nodes[bin_key] = (node_idx, y_val)

        # Extract node indices
        anterior_indices = [idx for idx, _ in anterior_nodes.values()]

        return np.array(anterior_indices, dtype=np.int32)

    def get_boundary_nodes(
        self,
        mesh_nodes: NDArray[np.float64],
    ) -> NDArray[np.int32]:
        """
        Identify mesh nodes that should have fixed boundary conditions.

        This includes:
        1. Skull boundary nodes (computed from dilated brain mask)
        2. Anterior brainstem boundary nodes (clivus - petrous bone)

        The anterior brainstem boundary represents the clivus, which acts as
        a hard constraint preventing anterior tumor expansion into the prepontine
        cistern region.

        Args:
            mesh_nodes: Node positions, shape (N, 3)

        Returns:
            Indices of boundary nodes
        """
        boundary_indices = set()
        skull = self.compute_skull_boundary()
        segmentation = self.load_tissue_segmentation()

        # Add skull boundary nodes
        for i, pos in enumerate(mesh_nodes):
            if self.is_skull_boundary(pos):
                boundary_indices.add(i)

        # Add anterior brainstem boundary (clivus)
        anterior_nodes = self._compute_anterior_brainstem_boundary(mesh_nodes)
        boundary_indices.update(anterior_nodes)

        return np.array(sorted(boundary_indices), dtype=np.int32)

    def assign_node_tissues(
        self,
        mesh_nodes: NDArray[np.float64],
    ) -> NDArray[np.int32]:
        """
        Assign tissue types to mesh nodes.

        Args:
            mesh_nodes: Node positions, shape (N, 3)

        Returns:
            Tissue labels for each node
        """
        segmentation = self.load_tissue_segmentation()
        labels = np.zeros(len(mesh_nodes), dtype=np.int32)

        for i, pos in enumerate(mesh_nodes):
            tissue = segmentation.get_tissue_at_point(pos)
            labels[i] = tissue.value

        return labels

    def get_fiber_directions_at_nodes(
        self,
        mesh_nodes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Get fiber orientation vectors at mesh nodes.

        Args:
            mesh_nodes: Node positions, shape (N, 3)

        Returns:
            Fiber direction vectors, shape (N, 3)
        """
        segmentation = self.load_tissue_segmentation()
        fibers = self.load_fiber_orientation()
        directions = np.zeros((len(mesh_nodes), 3), dtype=np.float64)

        for i, pos in enumerate(mesh_nodes):
            directions[i] = fibers.get_direction_at_point(
                pos, segmentation.voxel_size
            )

        return directions
