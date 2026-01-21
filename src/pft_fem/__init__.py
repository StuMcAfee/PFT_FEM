"""
PFT_FEM: Posterior Fossa Tumor Finite Element Modeling

A pipeline for simulating MRI images by modeling tumor growth in the
posterior cranial fossa using finite element methods, starting from
the SUIT cerebellar atlas.
"""

__version__ = "0.1.0"

from .atlas import SUITAtlasLoader, AtlasProcessor
from .mesh import MeshGenerator, TetMesh
from .fem import TumorGrowthSolver, MaterialProperties
from .simulation import MRISimulator, TumorParameters
from .io import NIfTIWriter, load_nifti, save_nifti

__all__ = [
    "SUITAtlasLoader",
    "AtlasProcessor",
    "MeshGenerator",
    "TetMesh",
    "TumorGrowthSolver",
    "MaterialProperties",
    "MRISimulator",
    "TumorParameters",
    "NIfTIWriter",
    "load_nifti",
    "save_nifti",
]
