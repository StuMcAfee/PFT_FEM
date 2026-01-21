"""
Pytest configuration and shared fixtures for PFT_FEM tests.
"""

import pytest
import numpy as np

# Add src to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def synthetic_atlas():
    """
    Create a synthetic SUIT atlas for testing.

    This fixture is session-scoped to avoid regenerating for each test.
    """
    from pft_fem.atlas import SUITAtlasLoader

    loader = SUITAtlasLoader(atlas_dir=None)  # Will generate synthetic
    return loader.load()


@pytest.fixture
def small_spherical_mask():
    """Small spherical mask for quick tests."""
    shape = (11, 11, 11)
    center = np.array([5, 5, 5])
    radius = 4

    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    return dist <= radius


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
