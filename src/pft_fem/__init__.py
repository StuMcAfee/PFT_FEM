"""
PFT_FEM: Posterior Fossa Tumor Finite Element Modeling

A pipeline for simulating MRI images by modeling tumor growth in the
posterior cranial fossa using finite element methods, starting from
the SUIT cerebellar atlas.

Includes biophysical constraints for realistic brain tissue modeling:
- White matter: Anisotropic, resists stretching along fiber direction
- Gray matter: Compressible uniformly, isotropic
- Skull boundary: Immovable constraint
- Tissue-specific diffusion: Enhanced along white matter tracts
"""

__version__ = "0.1.0"

from .atlas import SUITAtlasLoader, AtlasProcessor
from .mesh import MeshGenerator, TetMesh
from .fem import TumorGrowthSolver, MaterialProperties, TissueType
from .simulation import MRISimulator, TumorParameters
from .io import NIfTIWriter, load_nifti, save_nifti
from .biophysical_constraints import (
    BiophysicalConstraints,
    BrainTissue,
    TissueSegmentation,
    FiberOrientation,
    AnisotropicMaterialProperties,
    SUITPyIntegration,
    MNIAtlasLoader,
    SpaceTransformer,
    DEFAULT_TUMOR_ORIGIN_MNI,
    POSTERIOR_FOSSA_BOUNDS_MNI,
)

__all__ = [
    # Atlas loading
    "SUITAtlasLoader",
    "AtlasProcessor",
    # Mesh generation
    "MeshGenerator",
    "TetMesh",
    # FEM solver
    "TumorGrowthSolver",
    "MaterialProperties",
    "TissueType",
    # MRI simulation
    "MRISimulator",
    "TumorParameters",
    # I/O
    "NIfTIWriter",
    "load_nifti",
    "save_nifti",
    # Biophysical constraints
    "BiophysicalConstraints",
    "BrainTissue",
    "TissueSegmentation",
    "FiberOrientation",
    "AnisotropicMaterialProperties",
    "SUITPyIntegration",
    "MNIAtlasLoader",
    "SpaceTransformer",
    # Default parameters
    "DEFAULT_TUMOR_ORIGIN_MNI",
    "POSTERIOR_FOSSA_BOUNDS_MNI",
]
