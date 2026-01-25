# PFT_FEM: Posterior Fossa Tumor Finite Element Modeling

A scientific computing pipeline that simulates MRI images by modeling tumor growth in the posterior cranial fossa (cerebellum region) using finite element methods (FEM).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Detailed Pipeline Steps](#detailed-pipeline-steps)
  - [Step 1: Atlas Loading](#step-1-atlas-loading)
  - [Step 2: Mesh Generation](#step-2-mesh-generation)
  - [Step 3: Tumor Growth Simulation](#step-3-tumor-growth-simulation)
  - [Step 4: MRI Image Generation](#step-4-mri-image-generation)
  - [Step 5: Results Output](#step-5-results-output)
- [Command-Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Configuration Options](#configuration-options)
- [Output Files](#output-files)
- [Physical Models](#physical-models)
- [Examples](#examples)
- [Testing](#testing)
- [License](#license)

---

## Overview

PFT_FEM generates synthetic MRI data showing realistic tumor progression in the cerebellum. Starting from a SUIT (Spatially Unbiased Infratentorial Template) cerebellar atlas, it:

1. Loads anatomical brain atlas data
2. Generates a tetrahedral finite element mesh
3. Simulates tumor growth using reaction-diffusion equations
4. Models tissue deformation due to tumor mass effect
5. Produces synthetic MRI images in multiple sequences

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PFT_FEM PIPELINE OVERVIEW                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SUIT Atlas ──► Mesh Generation ──► FEM Simulation ──► MRI Synthesis      │
│                                                                             │
│   ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│   │ NIfTI   │    │ Tetrahedral │    │ Tumor Cell  │    │ T1, T2,     │     │
│   │ Volume  │───►│ Mesh        │───►│ Density +   │───►│ FLAIR, DWI  │     │
│   │ Labels  │    │ (Nodes +    │    │ Displacement│    │ Images      │     │
│   └─────────┘    │ Elements)   │    └─────────────┘    └─────────────┘     │
│                  └─────────────┘                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/PFT_FEM.git
cd PFT_FEM

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

After installation, verify that the CLI is available:

```bash
# Check that the command is installed
pft-simulate --version

# Or check via Python
python -c "import pft_fem; print(pft_fem.__version__)"
```

> **Note:** If you get "command not found" or "not recognized", make sure:
> 1. You ran `pip install -e .` from the project root (where `pyproject.toml` is located)
> 2. Your Python scripts directory is in your system PATH
> 3. You're using the same Python environment where you installed the package

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Numerical computing |
| scipy | >=1.7.0 | Scientific algorithms |
| nibabel | >=4.0.0 | NIfTI file I/O |
| meshio | >=5.0.0 | Mesh I/O formats |
| scikit-image | >=0.19.0 | Image processing |
| matplotlib | >=3.5.0 | Visualization |

---

## Quick Start

> **Important:** You must install the package first before using the CLI or Python API.
> Run `pip install -e .` from the project root directory.

### Command Line

After installation, the `pft-simulate` command will be available:

```bash
# Run with default parameters (synthetic atlas, 30-day simulation)
pft-simulate -o ./output -v

# Run with custom tumor parameters
pft-simulate -o ./results \
    --tumor-center 10 -5 0 \
    --tumor-radius 5.0 \
    --duration 45 \
    --sequences T1 T2 FLAIR \
    -v
```

### Python API

```python
from pft_fem import SUITAtlasLoader, MRISimulator, TumorParameters
from pft_fem.io import NIfTIWriter

# Load atlas (uses synthetic if no real atlas provided)
loader = SUITAtlasLoader()
atlas_data = loader.load()

# Configure tumor parameters
tumor_params = TumorParameters(
    center=(0.0, 0.0, 0.0),
    initial_radius=5.0,
    proliferation_rate=0.012,
    diffusion_rate=0.15
)

# Run simulation
simulator = MRISimulator(atlas_data, tumor_params)
result = simulator.run_full_pipeline(duration_days=30, verbose=True)

# Save results
writer = NIfTIWriter(output_dir="./output", affine=atlas_data.affine)
output_paths = writer.write_simulation_results(result)
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              COMPLETE PIPELINE FLOW                             │
└─────────────────────────────────────────────────────────────────────────────────┘

     INPUT                    PROCESSING                         OUTPUT
  ═══════════            ═══════════════════                 ═══════════════

  ┌───────────┐
  │ SUIT Atlas│          ┌─────────────────┐
  │ Directory │─────────►│  STEP 1: ATLAS  │
  │ (optional)│          │    LOADING      │
  └───────────┘          └────────┬────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │   AtlasData    │
                         │  ┌──────────┐  │
                         │  │ template │  │
                         │  │ labels   │  │
                         │  │ affine   │  │
                         │  │ regions  │  │
                         │  └──────────┘  │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
  ┌───────────┐          │  STEP 2: MESH   │
  │  Tissue   │─────────►│   GENERATION    │
  │   Mask    │          └────────┬────────┘
  └───────────┘                   │
                                  ▼
                         ┌────────────────┐
                         │    TetMesh     │
                         │  ┌──────────┐  │
                         │  │ nodes    │  │
                         │  │ elements │  │
                         │  │ labels   │  │
                         │  └──────────┘  │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
  ┌───────────┐          │ STEP 3: TUMOR   │
  │  Tumor    │─────────►│    GROWTH       │
  │  Params   │          │   SIMULATION    │
  └───────────┘          └────────┬────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │  TumorState[]  │
                         │  ┌──────────┐  │
                         │  │ density  │  │
                         │  │ displace │  │
                         │  │ stress   │  │
                         │  │ time     │  │
                         │  └──────────┘  │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
  ┌───────────┐          │  STEP 4: MRI    │
  │   MRI     │─────────►│    IMAGE        │
  │ Sequence  │          │   GENERATION    │
  │  Params   │          └────────┬────────┘
  └───────────┘                   │
                                  ▼
                         ┌────────────────┐
                         │ SimulationResult│
                         │  ┌──────────┐  │
                         │  │ mri_vols │  │
                         │  │ masks    │  │
                         │  │ metadata │  │
                         │  └──────────┘  │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌─────────────────┐          ┌─────────────────┐
                         │  STEP 5: FILE   │─────────►│  NIfTI Files    │
                         │     OUTPUT      │          │  + JSON         │
                         └─────────────────┘          └─────────────────┘
```

---

## Detailed Pipeline Steps

### Step 1: Atlas Loading

The pipeline begins by loading anatomical reference data from the SUIT cerebellar atlas.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: ATLAS LOADING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT FILES (Optional - uses synthetic if not provided)                    │
│  ════════════════════════════════════════════════════════                   │
│                                                                             │
│  atlas_directory/                                                           │
│  ├── SUIT_template.nii.gz    ◄── T1-weighted template image                 │
│  └── SUIT_labels.nii.gz      ◄── Integer label volume (1-30)                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROCESSING                                                                 │
│  ══════════                                                                 │
│                                                                             │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐     │
│  │ Load NIfTI or   │      │ Extract affine  │      │ Build region    │     │
│  │ generate        │─────►│ transformation  │─────►│ dictionary      │     │
│  │ synthetic       │      │ matrix          │      │ (label→name)    │     │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OUTPUT: AtlasData                                                          │
│  ═════════════════                                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ AtlasData                                                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ template:   NDArray[float32]  │ Shape: (91, 109, 91)                │   │
│  │ labels:     NDArray[int32]    │ Values: 0-30                        │   │
│  │ affine:     NDArray[float64]  │ 4x4 transformation matrix           │   │
│  │ voxel_size: tuple[float, ...] │ (2.0, 2.0, 2.0) mm                  │   │
│  │ regions:    dict[int, str]    │ {1: "Left_I_IV", 2: "Right_I_IV"...}│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VISUALIZATION: Atlas Label Map (Axial Slice)                               │
│  ════════════════════════════════════════════                               │
│                                                                             │
│           Anterior                                                          │
│              ▲                                                              │
│              │                                                              │
│    ┌─────────────────────┐                                                  │
│    │ ░░░░░░░░░░░░░░░░░░░ │  ░ = Background (0)                              │
│    │ ░░░░░▓▓▓▓▓▓▓░░░░░░░ │  ▓ = Vermis (Labels 15-28)                       │
│    │ ░░░▒▒▒▓▓▓▓▓▒▒▒░░░░░ │  ▒ = Left Hemisphere (Labels 1-14)               │
│    │ ░░▒▒▒▒▒▓▓▓▒▒▒▒▒░░░░ │  ▓ = Right Hemisphere (Labels 1-14)              │
│    │ ░░▒▒▒▒▒███▒▒▒▒▒░░░░ │  █ = Brainstem (Label 29)                        │
│    │ ░░░▒▒▒▒███▒▒▒▒░░░░░ │  ■ = Fourth Ventricle (Label 30)                 │
│    │ ░░░░░░░███░░░░░░░░░ │                                                  │
│    │ ░░░░░░░░░░░░░░░░░░░ │                                                  │
│    └─────────────────────┘                                                  │
│  Left ◄─────────────────► Right                                             │
│              │                                                              │
│              ▼                                                              │
│          Posterior                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code Example:**
```python
from pft_fem import SUITAtlasLoader

# Option 1: Load real SUIT atlas
loader = SUITAtlasLoader(atlas_dir="/path/to/suit/atlas")
atlas_data = loader.load()

# Option 2: Use synthetic atlas (default)
loader = SUITAtlasLoader()
atlas_data = loader.load()

print(f"Atlas shape: {atlas_data.shape}")
print(f"Voxel size: {atlas_data.voxel_size} mm")
print(f"Number of regions: {len(atlas_data.regions)}")
```

**Region Labels:**
| Label | Region | Label | Region |
|-------|--------|-------|--------|
| 1-14 | Left cerebellar lobules | 15-28 | Right cerebellar lobules |
| 29 | Brainstem | 30 | Fourth Ventricle |

---

### Step 2: Mesh Generation

Converts the volumetric atlas into a tetrahedral finite element mesh.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STEP 2: MESH GENERATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                                                                      │
│  ═════                                                                      │
│                                                                             │
│  ┌──────────────┐    ┌───────────────┐                                      │
│  │  AtlasData   │    │  Tissue Mask  │   Binary mask of tissue region       │
│  │  (from Step 1)│    │  (extracted)  │   to be meshed                       │
│  └──────────────┘    └───────────────┘                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROCESSING: Voxel-to-Tetrahedra Conversion                                 │
│  ══════════════════════════════════════════                                 │
│                                                                             │
│   Each voxel is subdivided into 5-6 tetrahedra:                             │
│                                                                             │
│        Voxel                    Tetrahedra                                  │
│      ┌────────┐               ┌────────┐                                    │
│     /│       /│              /│\      /│                                    │
│    / │      / │             / │ \    / │                                    │
│   ┌────────┐  │     ───►   ┌──│──\──┐  │                                    │
│   │  │     │  │            │\ │   \ │  │                                    │
│   │  └─────│──┘            │ \│    \│──┘                                    │
│   │ /      │ /             │  └─────│─/                                     │
│   │/       │/              │ /  \   │/                                      │
│   └────────┘               └/────\──┘                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OUTPUT: TetMesh                                                            │
│  ══════════════                                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TetMesh                                                             │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ nodes:          NDArray[float64]  │ Shape: (N, 3) - XYZ coords      │   │
│  │ elements:       NDArray[int32]    │ Shape: (M, 4) - Node indices    │   │
│  │ node_labels:    NDArray[int32]    │ Shape: (N,) - Tissue type       │   │
│  │ boundary_nodes: NDArray[int32]    │ Surface node indices            │   │
│  │ node_neighbors: list[set[int]]    │ Adjacency information           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VISUALIZATION: Tetrahedral Mesh (2D Cross-Section)                         │
│  ══════════════════════════════════════════════════                         │
│                                                                             │
│  Original Volume Mask          Resulting Mesh                               │
│  ┌───────────────────┐        ┌───────────────────┐                         │
│  │ ░░░░░░░░░░░░░░░░░ │        │                   │                         │
│  │ ░░░░████████░░░░░ │        │    ●───●───●───●  │     ● = Node            │
│  │ ░░██████████████░ │        │   /│\ /│\ /│\ /│  │     ─ = Edge            │
│  │ ░████████████████ │  ───►  │  ●─┼─●─┼─●─┼─●─┼  │     △ = Element face   │
│  │ ░████████████████ │        │   \│/ \│/ \│/ \│  │                         │
│  │ ░░██████████████░ │        │    ●───●───●───●  │                         │
│  │ ░░░░████████░░░░░ │        │     \ /│\ /│\ /   │                         │
│  │ ░░░░░░░░░░░░░░░░░ │        │      ●─┼─●─┼─●    │                         │
│  └───────────────────┘        └───────────────────┘                         │
│                                                                             │
│  Mesh Statistics (typical):                                                 │
│  • Nodes: ~50,000-200,000                                                   │
│  • Elements: ~250,000-1,000,000                                             │
│  • Element quality: >0.1 (aspect ratio)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code Example:**
```python
from pft_fem import MeshGenerator, AtlasProcessor

# Get tissue mask from atlas
processor = AtlasProcessor(atlas_data)
tissue_mask = processor.get_tissue_mask("cerebellum")

# Generate mesh
generator = MeshGenerator()
mesh = generator.from_mask(
    mask=tissue_mask,
    voxel_size=atlas_data.voxel_size,
    labels=atlas_data.labels,
    affine=atlas_data.affine,
    simplify=True
)

# Check mesh quality
quality = mesh.compute_quality_metrics()
print(f"Number of nodes: {len(mesh.nodes)}")
print(f"Number of elements: {len(mesh.elements)}")
print(f"Mean element quality: {quality['mean_quality']:.3f}")
```

---

### Step 3: Tumor Growth Simulation

Simulates tumor evolution using coupled reaction-diffusion and mechanical equations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: TUMOR GROWTH SIMULATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                                                                      │
│  ═════                                                                      │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │ TumorParameters                                                   │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │ center:               (x, y, z)  │ Tumor seed location in mm      │     │
│  │ initial_radius:       float      │ Starting radius in mm          │     │
│  │ proliferation_rate:   float      │ Growth rate (1/day)            │     │
│  │ diffusion_rate: float     │ Spread rate (mm²/day)          │     │
│  │ carrying_capacity:    float      │ Max cell density (normalized)  │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROCESSING: Operator Splitting FEM                                         │
│  ══════════════════════════════════                                         │
│                                                                             │
│  For each time step Δt:                                                     │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Diffusion  │    │  Reaction   │    │  Compute    │    │  Solve      │  │
│  │  (Implicit  │───►│  (Explicit  │───►│  Growth     │───►│  Mechanical │  │
│  │   Euler)    │    │   Euler)    │    │  Force      │    │  Equilib.   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │           │
│        ▼                  ▼                  ▼                  ▼           │
│   ∂c/∂t = D∇²c      c += ρc(1-c/K)Δt    F = σ_growth      Ku = F           │
│                                                                             │
│  where:                                                                     │
│    c = tumor cell density                                                   │
│    D = diffusion coefficient                                                │
│    ρ = proliferation rate                                                   │
│    K = carrying capacity                                                    │
│    F = growth-induced force                                                 │
│    u = displacement field                                                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VISUALIZATION: Tumor Evolution Over Time                                   │
│  ════════════════════════════════════════                                   │
│                                                                             │
│  Day 0 (Initial)      Day 15              Day 30 (Final)                    │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
│  │               │   │               │   │               │                  │
│  │      ▪        │   │     ███       │   │   ███████     │                  │
│  │      ▪        │   │    █████      │   │  █████████    │                  │
│  │               │   │     ███       │   │   ███████     │                  │
│  │               │   │               │   │               │                  │
│  └───────────────┘   └───────────────┘   └───────────────┘                  │
│                                                                             │
│  Cell Density:  ▪ seed (0.8)  █ high (>0.5)  ░ low (0.1-0.5)                │
│                                                                             │
│  Tissue Displacement (arrows show direction):                               │
│  ┌───────────────────────────────────────┐                                  │
│  │           ↑     ↑     ↑               │                                  │
│  │        ↖  ↑  ███████  ↑  ↗            │                                  │
│  │      ←    ← █████████ →    →          │   Tumor mass pushes              │
│  │        ↙  ↓  ███████  ↓  ↘            │   surrounding tissue             │
│  │           ↓     ↓     ↓               │   outward                        │
│  └───────────────────────────────────────┘                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OUTPUT: TumorState (list for each time step)                               │
│  ════════════════════════════════════════════                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TumorState                                                          │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ cell_density:  NDArray[float64]  │ Shape: (N,) - density at nodes   │   │
│  │ displacement:  NDArray[float64]  │ Shape: (N, 3) - XYZ displacement │   │
│  │ stress:        NDArray[float64]  │ Shape: (M, 6) - stress tensor    │   │
│  │ time:          float             │ Simulation time in days          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code Example:**
```python
from pft_fem import TumorGrowthSolver, TumorState, MaterialProperties

# Initialize solver
material = MaterialProperties(
    young_modulus=3000.0,      # Pa
    poisson_ratio=0.45,
    proliferation_rate=0.012,  # 1/day
    diffusion_coefficient=0.15 # mm²/day
)
solver = TumorGrowthSolver(mesh, material)

# Create initial tumor state
initial_state = TumorState.initial(
    mesh=mesh,
    seed_center=(0.0, 0.0, 0.0),
    seed_radius=5.0,
    seed_density=0.8
)

# Run simulation
states = solver.simulate(
    initial_state=initial_state,
    duration=30.0,  # days
    dt=1.0,         # time step
    callback=lambda s: print(f"Day {s.time}: Volume = {solver.compute_tumor_volume(s):.1f} mm³")
)

# Get final state
final_state = states[-1]
print(f"Final tumor volume: {solver.compute_tumor_volume(final_state):.1f} mm³")
print(f"Max displacement: {solver.compute_max_displacement(final_state):.2f} mm")
```

---

### Step 4: MRI Image Generation

Generates synthetic MRI images from the tumor simulation results.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 4: MRI IMAGE GENERATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                                                                      │
│  ═════                                                                      │
│                                                                             │
│  ┌──────────────┐    ┌───────────────────────────────────────────────┐     │
│  │ TumorState   │    │ MRI Sequence Parameters                       │     │
│  │ (from Step 3)│    │  • Sequence type (T1, T2, FLAIR, etc.)        │     │
│  └──────────────┘    │  • TR (Repetition Time)                       │     │
│                      │  • TE (Echo Time)                             │     │
│                      │  • TI (Inversion Time, for FLAIR)             │     │
│                      └───────────────────────────────────────────────┘     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROCESSING                                                                 │
│  ══════════                                                                 │
│                                                                             │
│  1. Create Tissue Maps                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Cell Density ──► Threshold ──► Tumor Mask                          │   │
│  │       │                              │                              │   │
│  │       └──► Gradient ──► Edema Mask ──┘                              │   │
│  │                              │                                      │   │
│  │  Atlas Labels ──────────────┴──► Combined Tissue Map                │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. Compute MRI Signal                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  For each tissue type:                                              │   │
│  │                                                                     │   │
│  │    Signal = f(T1, T2, PD, TR, TE, TI)                               │   │
│  │                                                                     │   │
│  │  Tissue Relaxation Times (ms):                                      │   │
│  │  ┌──────────────┬────────┬────────┬────────┐                        │   │
│  │  │ Tissue       │   T1   │   T2   │   PD   │                        │   │
│  │  ├──────────────┼────────┼────────┼────────┤                        │   │
│  │  │ Gray Matter  │  1200  │   80   │  0.95  │                        │   │
│  │  │ White Matter │   800  │   70   │  0.85  │                        │   │
│  │  │ CSF          │  4000  │  2000  │  1.00  │                        │   │
│  │  │ Tumor        │  1400  │  100   │  0.98  │                        │   │
│  │  │ Edema        │  1500  │  120   │  0.96  │                        │   │
│  │  │ Necrosis     │  1000  │   60   │  0.75  │                        │   │
│  │  └──────────────┴────────┴────────┴────────┘                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. Apply Deformation                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Displacement Field ──► Interpolate ──► Deformed MRI Image          │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VISUALIZATION: MRI Sequences Comparison                                    │
│  ═══════════════════════════════════════                                    │
│                                                                             │
│       T1-weighted          T2-weighted           FLAIR                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │              │
│  │ ▓▓▓░░░░░░░░▓▓▓▓ │  │ ▒▒▒████████▒▒▒▒ │  │ ▓▓▓░░░░░░░░▓▓▓▓ │              │
│  │ ▓▓░░░░██░░░░▓▓▓ │  │ ▒▒████░░████▒▒▒ │  │ ▓▓░░░░██░░░░▓▓▓ │              │
│  │ ▓░░░░████░░░░▓▓ │  │ ▒████░░░░████▒▒ │  │ ▓░░░████░░░░░▓▓ │              │
│  │ ▓░░░░████░░░░▓▓ │  │ ▒████░░░░████▒▒ │  │ ▓░░░████░░░░░▓▓ │              │
│  │ ▓▓░░░░██░░░░▓▓▓ │  │ ▒▒████░░████▒▒▒ │  │ ▓▓░░░░██░░░░▓▓▓ │              │
│  │ ▓▓▓░░░░░░░░▓▓▓▓ │  │ ▒▒▒████████▒▒▒▒ │  │ ▓▓▓░░░░░░░░▓▓▓▓ │              │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                             │
│  Legend:  ▓ Gray Matter   ░ White Matter   █ Tumor   ▒ CSF (bright T2)      │
│                                                                             │
│  Key Characteristics:                                                       │
│  • T1: Gray matter darker than white matter, tumor iso/hypointense          │
│  • T2: CSF bright, tumor bright with dark necrotic core                     │
│  • FLAIR: CSF suppressed (dark), edema appears bright                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OUTPUT: Dictionary of MRI Volumes                                          │
│  ═════════════════════════════════                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ mri_images: dict[MRISequence, NDArray[float32]]                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ "T1":         NDArray  │ Shape: (91, 109, 91)                       │   │
│  │ "T2":         NDArray  │ Shape: (91, 109, 91)                       │   │
│  │ "FLAIR":      NDArray  │ Shape: (91, 109, 91)                       │   │
│  │ "T1_contrast": NDArray │ Shape: (91, 109, 91) (optional)            │   │
│  │ "DWI":        NDArray  │ Shape: (91, 109, 91) (optional)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code Example:**
```python
from pft_fem import MRISimulator, MRISequence

# Generate MRI images
mri_images = simulator.generate_mri(
    tumor_state=final_state,
    sequences=[MRISequence.T1, MRISequence.T2, MRISequence.FLAIR],
    TR=2000,   # Repetition time in ms
    TE=30,     # Echo time in ms
    TI=2500    # Inversion time in ms (for FLAIR)
)

# Access individual sequences
t1_image = mri_images[MRISequence.T1]
t2_image = mri_images[MRISequence.T2]

print(f"T1 image shape: {t1_image.shape}")
print(f"T1 intensity range: [{t1_image.min():.2f}, {t1_image.max():.2f}]")
```

---

### Step 5: Results Output

Writes all simulation results to NIfTI format files.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 5: RESULTS OUTPUT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                                                                      │
│  ═════                                                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ SimulationResult                                                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ mri_images:     dict[str, NDArray]   │ MRI volumes by sequence      │   │
│  │ tumor_mask:     NDArray[bool]        │ Binary tumor segmentation    │   │
│  │ edema_mask:     NDArray[bool]        │ Binary edema segmentation    │   │
│  │ deformed_atlas: NDArray[int32]       │ Deformed label volume        │   │
│  │ displacement:   NDArray[float64]     │ 3D displacement field        │   │
│  │ metadata:       dict                 │ Simulation parameters        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROCESSING                                                                 │
│  ══════════                                                                 │
│                                                                             │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │ Convert arrays   │────►│ Create NIfTI     │────►│ Write .nii.gz    │    │
│  │ to proper dtype  │     │ headers          │     │ files            │    │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OUTPUT: File Structure                                                     │
│  ══════════════════════                                                     │
│                                                                             │
│  output_directory/                                                          │
│  │                                                                          │
│  ├── pft_simulation_mri_T1.nii.gz           ◄── T1-weighted image           │
│  ├── pft_simulation_mri_T2.nii.gz           ◄── T2-weighted image           │
│  ├── pft_simulation_mri_FLAIR.nii.gz        ◄── FLAIR image                 │
│  ├── pft_simulation_mri_T1_contrast.nii.gz  ◄── Contrast-enhanced (opt.)    │
│  ├── pft_simulation_mri_DWI.nii.gz          ◄── Diffusion-weighted (opt.)   │
│  │                                                                          │
│  ├── pft_simulation_deformed_atlas.nii.gz   ◄── Deformed label volume       │
│  ├── pft_simulation_tumor_mask.nii.gz       ◄── Binary tumor mask           │
│  ├── pft_simulation_edema_mask.nii.gz       ◄── Binary edema mask           │
│  │                                                                          │
│  └── pft_simulation_metadata.json           ◄── Simulation metadata         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  METADATA JSON STRUCTURE                                                    │
│  ═══════════════════════                                                    │
│                                                                             │
│  {                                                                          │
│    "duration_days": 30.0,                                                   │
│    "final_tumor_volume_mm3": 2450.5,                                        │
│    "max_displacement_mm": 3.2,                                              │
│    "tumor_params": {                                                        │
│      "center": [0.0, 0.0, 0.0],                                             │
│      "initial_radius": 5.0,                                                 │
│      "proliferation_rate": 0.012,                                           │
│      "diffusion_rate": 0.15                                          │
│    },                                                                       │
│    "atlas_shape": [91, 109, 91],                                            │
│    "voxel_size": [2.0, 2.0, 2.0],                                           │
│    "sequences_generated": ["T1", "T2", "FLAIR"],                            │
│    "timestamp": "2024-01-15T10:30:00"                                       │
│  }                                                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VISUALIZATION: Output Files Overview                                       │
│  ════════════════════════════════════                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        OUTPUT DIRECTORY                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│       MRI Images              Segmentations           Metadata              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  ┌───┐ ┌───┐    │     │     ┌───┐       │     │ {               │       │
│  │  │T1 │ │T2 │    │     │     │ █ │       │     │   "duration":   │       │
│  │  └───┘ └───┘    │     │     │███│       │     │     30,         │       │
│  │  ┌───┐ ┌───┐    │     │     └───┘       │     │   "volume":     │       │
│  │  │FLR│ │DWI│    │     │   tumor_mask    │     │     2450.5      │       │
│  │  └───┘ └───┘    │     │                 │     │ }               │       │
│  │  .nii.gz files  │     │  .nii.gz files  │     │  .json file     │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code Example:**
```python
from pft_fem.io import NIfTIWriter

# Create writer
writer = NIfTIWriter(
    output_dir="./output",
    affine=atlas_data.affine,
    prefix="pft_simulation"
)

# Write all results
output_paths = writer.write_simulation_results(result)

# Print generated files
for name, path in output_paths.items():
    print(f"{name}: {path}")
```

---

## Command-Line Interface

The `pft-simulate` command provides a convenient way to run simulations.

```
Usage: pft-simulate [OPTIONS]

Options:
  -o, --output PATH          Output directory (default: ./output)
  -a, --atlas PATH           Path to SUIT atlas directory (optional)
  -d, --duration FLOAT       Simulation duration in days (default: 30)
  --tumor-center X Y Z       Tumor seed center in mm (default: 0 0 0)
  --tumor-radius FLOAT       Initial tumor radius in mm (default: 2)
  --proliferation-rate FLOAT Growth rate in 1/day (default: 0.012)
  --diffusion-rate FLOAT     Diffusion rate in mm²/day (default: 0.15)
  --sequences [T1|T2|FLAIR|T1_contrast|DWI]
                             MRI sequences to generate (default: T1 T2 FLAIR)
  -v, --verbose              Enable verbose output
  --version                  Show version and exit
  -h, --help                 Show this message and exit
```

### Examples

```bash
# Basic simulation with defaults
pft-simulate -o ./results

# Custom tumor location and size
pft-simulate -o ./results \
    --tumor-center 10 -5 0 \
    --tumor-radius 8.0 \
    --duration 60

# Generate all MRI sequences with verbose output
pft-simulate -o ./results \
    --sequences T1 T2 FLAIR T1_contrast DWI \
    -v

# Use real SUIT atlas
pft-simulate -o ./results \
    -a /path/to/suit/atlas \
    --tumor-radius 5.0
```

---

## Python API

### Complete Pipeline Example

```python
from pft_fem import (
    SUITAtlasLoader,
    AtlasProcessor,
    MeshGenerator,
    TumorGrowthSolver,
    TumorState,
    MaterialProperties,
    MRISimulator,
    TumorParameters,
    MRISequence
)
from pft_fem.io import NIfTIWriter

# Step 1: Load Atlas
loader = SUITAtlasLoader()
atlas_data = loader.load()
print(f"Loaded atlas: {atlas_data.shape}")

# Step 2: Generate Mesh
processor = AtlasProcessor(atlas_data)
tissue_mask = processor.get_tissue_mask("all")
generator = MeshGenerator()
mesh = generator.from_mask(
    mask=tissue_mask,
    voxel_size=atlas_data.voxel_size,
    labels=atlas_data.labels,
    affine=atlas_data.affine
)
print(f"Generated mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")

# Step 3: Run Tumor Simulation
material = MaterialProperties(
    young_modulus=3000.0,
    poisson_ratio=0.45,
    proliferation_rate=0.012,
    diffusion_coefficient=0.15
)
solver = TumorGrowthSolver(mesh, material)

initial_state = TumorState.initial(
    mesh=mesh,
    seed_center=(0.0, 0.0, 0.0),
    seed_radius=5.0,
    seed_density=0.8
)

states = solver.simulate(
    initial_state=initial_state,
    duration=30.0,
    dt=1.0,
    callback=lambda s: print(f"Day {s.time:.0f}")
)

final_state = states[-1]
print(f"Final tumor volume: {solver.compute_tumor_volume(final_state):.1f} mm³")

# Step 4: Generate MRI Images
tumor_params = TumorParameters(center=(0.0, 0.0, 0.0), initial_radius=5.0)
simulator = MRISimulator(atlas_data, tumor_params)
simulator.mesh = mesh
simulator.solver = solver

mri_images = simulator.generate_mri(
    tumor_state=final_state,
    sequences=[MRISequence.T1, MRISequence.T2, MRISequence.FLAIR]
)

# Step 5: Save Results
writer = NIfTIWriter(output_dir="./output", affine=atlas_data.affine)
# ... save results
```

### Using the High-Level API

```python
from pft_fem import SUITAtlasLoader, MRISimulator, TumorParameters

# Configure and run in one call
loader = SUITAtlasLoader()
atlas_data = loader.load()

tumor_params = TumorParameters(
    center=(5.0, -3.0, 0.0),
    initial_radius=6.0,
    proliferation_rate=0.015,
    diffusion_rate=0.2
)

simulator = MRISimulator(atlas_data, tumor_params)
result = simulator.run_full_pipeline(
    duration_days=45,
    sequences=[MRISequence.T1, MRISequence.T2, MRISequence.FLAIR],
    verbose=True
)

print(f"Generated {len(result.mri_images)} MRI sequences")
```

---

## Configuration Options

### Tumor Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `center` | (0, 0, 0) | Within atlas bounds | Tumor seed location (mm) |
| `initial_radius` | 2.5 | 1.0 - 20.0 | Starting radius (mm) |
| `proliferation_rate` | 0.012 | 0.001 - 0.1 | Growth rate (1/day) |
| `diffusion_rate` | 0.15 | 0.01 - 1.0 | Spread rate (mm²/day) |
| `carrying_capacity` | 1.0 | 0.5 - 1.0 | Max cell density |

### Material Properties

| Parameter | Default | Description |
|-----------|---------|-------------|
| `youngs_modulus` | 3000 Pa | Brain tissue stiffness |
| `poisson_ratio` | 0.45 | Near-incompressible |
| `growth_stress_coeff` | 0.1 Pa | Stress per unit density |

### MRI Sequence Parameters

| Sequence | TR (ms) | TE (ms) | TI (ms) | Notes |
|----------|---------|---------|---------|-------|
| T1 | 500-2000 | 10-30 | - | Anatomical detail |
| T2 | 2000-5000 | 80-120 | - | Fluid bright |
| FLAIR | 8000-11000 | 80-120 | 2500 | CSF suppressed |
| T1_contrast | 500-2000 | 10-30 | - | Enhanced tumor |
| DWI | 5000-10000 | 80-100 | - | Diffusion restriction |

---

## Output Files

### NIfTI Files (.nii.gz)

| File | Type | Description |
|------|------|-------------|
| `*_mri_T1.nii.gz` | float32 | T1-weighted MRI |
| `*_mri_T2.nii.gz` | float32 | T2-weighted MRI |
| `*_mri_FLAIR.nii.gz` | float32 | FLAIR sequence |
| `*_mri_T1_contrast.nii.gz` | float32 | Contrast-enhanced T1 |
| `*_mri_DWI.nii.gz` | float32 | Diffusion-weighted |
| `*_deformed_atlas.nii.gz` | int32 | Deformed label volume |
| `*_tumor_mask.nii.gz` | uint8 | Binary tumor mask |
| `*_edema_mask.nii.gz` | uint8 | Binary edema mask |

### Metadata JSON

```json
{
  "duration_days": 30.0,
  "final_tumor_volume_mm3": 2450.5,
  "max_displacement_mm": 3.2,
  "tumor_params": {
    "center": [0.0, 0.0, 0.0],
    "initial_radius": 5.0,
    "proliferation_rate": 0.012,
    "diffusion_rate": 0.15
  },
  "atlas_shape": [91, 109, 91],
  "voxel_size": [2.0, 2.0, 2.0]
}
```

---

## Biophysical Constraints

The simulation incorporates realistic biophysical constraints based on literature values for brain tissue mechanics and tumor biology.

### Tissue-Specific Material Properties

Different brain tissues have distinct mechanical and diffusion properties:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TISSUE-SPECIFIC BIOPHYSICAL PARAMETERS                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MECHANICAL PROPERTIES (Stiffness Multipliers)                              │
│  ═════════════════════════════════════════════                              │
│                                                                             │
│  Tissue Type      │ Stiffness │ Effective E │ Biological Basis              │
│  ─────────────────┼───────────┼─────────────┼─────────────────────────────  │
│  Gray Matter      │    1.0×   │   3000 Pa   │ Baseline brain tissue         │
│  White Matter     │    1.2×   │   3600 Pa   │ Fiber tracts add stiffness    │
│  CSF              │    0.01×  │     30 Pa   │ Fluid (nearly incompressible) │
│  Tumor            │    2.0×   │   6000 Pa   │ Dense cell packing            │
│  Edema            │    0.5×   │   1500 Pa   │ Fluid accumulation softens    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DIFFUSION PROPERTIES (Migration Multipliers)                               │
│  ════════════════════════════════════════════                               │
│                                                                             │
│  Tissue Type      │ Diffusion │ Effective D │ Biological Basis              │
│  ─────────────────┼───────────┼─────────────┼─────────────────────────────  │
│  Gray Matter      │    1.0×   │ 0.15 mm²/d  │ Baseline infiltration         │
│  White Matter     │    2.0×   │ 0.30 mm²/d  │ Faster along fiber tracts     │
│  CSF              │    0.1×   │ 0.015 mm²/d │ Barrier to tumor invasion     │
│  Tumor            │    0.5×   │ 0.075 mm²/d │ Dense tissue slows spread     │
│  Edema            │    1.5×   │ 0.225 mm²/d │ Loosened ECM aids migration   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Base Material Parameters

| Parameter | Default Value | Units | Description |
|-----------|---------------|-------|-------------|
| Young's Modulus (E) | 3000 | Pa | Brain tissue stiffness |
| Poisson Ratio (ν) | 0.45 | - | Nearly incompressible |
| Proliferation Rate (ρ) | 0.012 | 1/day | Cell division rate |
| Diffusion Coefficient (D) | 0.15 | mm²/day | Cell migration rate |
| Carrying Capacity (K) | 1.0 | - | Maximum cell density |
| Growth Stress Coefficient | 0.1 | Pa | Stress per unit tumor density |

### Tumor Region Classification

The simulation automatically classifies tumor regions based on cell density:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TUMOR REGION CLASSIFICATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        Cross-Section View                                   │
│                                                                             │
│                    ┌─────────────────────────┐                              │
│                    │ ░░░░░░░░░░░░░░░░░░░░░░░ │                              │
│                    │ ░░░░░▒▒▒▒▒▒▒▒▒▒▒░░░░░░░ │  ░ Normal tissue            │
│                    │ ░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░ │  ▒ Peritumoral edema        │
│                    │ ░░▒▒▒▒▓▓▓▓▓▓▓▓▓▒▒▒▒░░░░ │    (density < 0.1)          │
│                    │ ░░▒▒▒▓▓▓▓▓▓▓▓▓▓▓▒▒▒░░░░ │  ▓ Enhancing rim            │
│                    │ ░░▒▒▓▓▓▓███████▓▓▓▒▒░░░ │    (0.1 < density ≤ 0.5)    │
│                    │ ░░▒▒▓▓▓███████▓▓▓▒▒▒░░░ │  █ Tumor core               │
│                    │ ░░▒▒▒▓▓▓▓▓▓▓▓▓▓▓▒▒▒░░░░ │    (density > 0.5)          │
│                    │ ░░░▒▒▒▒▓▓▓▓▓▓▓▒▒▒▒░░░░░ │  ■ Necrotic center          │
│                    │ ░░░░░▒▒▒▒▒▒▒▒▒▒▒░░░░░░░ │    (density > 0.9)          │
│                    │ ░░░░░░░░░░░░░░░░░░░░░░░ │                              │
│                    └─────────────────────────┘                              │
│                                                                             │
│  Region           │ Density Threshold │ MRI Characteristics                 │
│  ─────────────────┼───────────────────┼───────────────────────────────────  │
│  Necrotic Core    │ > 0.9             │ T1 hypo, T2 hyper, no enhancement   │
│  Tumor Core       │ > 0.5             │ T1 iso/hypo, T2 hyper               │
│  Enhancing Rim    │ 0.1 - 0.5         │ Strong contrast enhancement         │
│  Edema            │ < 0.1 (dilated)   │ T2/FLAIR hyperintense               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Spatial Transforms

The simulation generates spatial transform outputs that describe how the tumor deforms surrounding tissue.

### Displacement Field

The 3D displacement field `u(x,y,z)` represents how each point in the brain moves due to tumor mass effect:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DISPLACEMENT FIELD OUTPUT                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Structure: 4D NIfTI volume (X × Y × Z × 3)                                 │
│  Units: millimeters (mm)                                                    │
│  Components: [dx, dy, dz] at each voxel                                     │
│                                                                             │
│  Visualization (2D slice):                                                  │
│                                                                             │
│         Before Tumor              After Tumor Growth                        │
│    ┌─────────────────────┐    ┌─────────────────────┐                       │
│    │ · · · · · · · · · · │    │ · · · · · · · · · · │                       │
│    │ · · · · · · · · · · │    │ · · ↖ ↑ ↑ ↑ ↗ · · · │                       │
│    │ · · · · · · · · · · │    │ · ← ↖ ↑ ↑ ↑ ↗ → · · │                       │
│    │ · · · · · · · · · · │    │ · ← ← █████ → → · · │   █ = Tumor           │
│    │ · · · · · · · · · · │ ─► │ · ← ← █████ → → · · │   → = Displacement    │
│    │ · · · · · · · · · · │    │ · ← ↙ ↓ ↓ ↓ ↘ → · · │       vectors         │
│    │ · · · · · · · · · · │    │ · · ↙ ↓ ↓ ↓ ↘ · · · │                       │
│    │ · · · · · · · · · · │    │ · · · · · · · · · · │                       │
│    └─────────────────────┘    └─────────────────────┘                       │
│                                                                             │
│  Key Metrics:                                                               │
│  • Max displacement: typically 1-5 mm for moderate tumors                   │
│  • Displacement decays with distance from tumor                             │
│  • Boundary conditions: fixed at domain edges                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Deformed Atlas

The deformed atlas applies the displacement field to the original anatomical labels:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEFORMED ATLAS OUTPUT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     Original Atlas              Deformed Atlas                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                         │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │                         │
│  │ ▓▓▓░░░░░░░░░░░▓▓▓▓▓ │    │ ▓▓▓░░░░░░░░░░░▓▓▓▓▓ │                         │
│  │ ▓▓░░░░░░░░░░░░░▓▓▓▓ │    │ ▓░░░░░░   ░░░░░░▓▓▓ │   Tumor pushes          │
│  │ ▓░░░░░░░░░░░░░░░▓▓▓ │ ─► │ ░░░░░░ ███ ░░░░░░▓▓ │   tissue outward        │
│  │ ▓░░░░░░░░░░░░░░░▓▓▓ │    │ ░░░░░░ ███ ░░░░░░▓▓ │                         │
│  │ ▓▓░░░░░░░░░░░░░▓▓▓▓ │    │ ▓░░░░░░   ░░░░░░▓▓▓ │   ░ = Cerebellum        │
│  │ ▓▓▓░░░░░░░░░░░▓▓▓▓▓ │    │ ▓▓▓░░░░░░░░░░░▓▓▓▓▓ │   ▓ = Brainstem         │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │   █ = Tumor             │
│  └─────────────────────┘    └─────────────────────┘                         │
│                                                                             │
│  Application: Track how tumor growth affects anatomical structures          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Coordinate Transforms

The simulation uses affine transformations to convert between coordinate systems:

| Transform | Description | Formula |
|-----------|-------------|---------|
| Voxel → Physical | Convert voxel indices to mm | `p = A × v` |
| Physical → Voxel | Convert mm to voxel indices | `v = A⁻¹ × p` |
| Apply Deformation | Warp image by displacement | `I'(x) = I(x - u(x))` |

**Output Files:**

| File | Format | Description |
|------|--------|-------------|
| `*_displacement.nii.gz` | float32 (X,Y,Z,3) | 3D displacement field in mm |
| `*_deformed_atlas.nii.gz` | float32 | Atlas with deformation applied |
| `*_jacobian.nii.gz` | float32 | Jacobian determinant (volume change) |

---

## Physical Models

### Reaction-Diffusion Equation (Tumor Growth)

The tumor cell density `c(x,t)` evolves according to the Fisher-Kolmogorov equation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    ∂c/∂t = D∇²c + ρc(1 - c/K)                               │
│                            ─────   ───────────                              │
│                           Diffusion   Logistic                              │
│                            (spread)   Growth                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Where:
- `c` = tumor cell density (0 to 1, normalized)
- `D` = diffusion coefficient (mm²/day) — varies by tissue type
- `ρ` = proliferation rate (1/day)
- `K` = carrying capacity (normalized to 1.0)

**Numerical Method:** Operator splitting with implicit Euler for diffusion, explicit Euler for reaction.

### Linear Elasticity (Tissue Deformation)

The displacement field `u(x)` satisfies the static equilibrium equation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                           ∇·σ + f = 0                                       │
│                                                                             │
│  where:                                                                     │
│    σ = λ(∇·u)I + μ(∇u + ∇uᵀ)     (Cauchy stress tensor)                     │
│    f = α · c · ∇c                 (growth-induced body force)               │
│                                                                             │
│  Lamé parameters from Young's modulus E and Poisson ratio ν:                │
│    λ = Eν / ((1+ν)(1-2ν))                                                   │
│    μ = E / (2(1+ν))                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Boundary Conditions:** Fixed (Dirichlet) at domain boundaries — displacement = 0.

**Numerical Method:** FEM with linear tetrahedral elements, conjugate gradient solver.

### MRI Signal Model

For spin-echo sequences:

```
S = PD × (1 - e^(-TR/T1)) × e^(-TE/T2)
```

For FLAIR (inversion recovery):

```
S = PD × |1 - 2e^(-TI/T1) + e^(-TR/T1)| × e^(-TE/T2)
```

**Tissue Relaxation Times (at 1.5T):**

| Tissue | T1 (ms) | T2 (ms) | PD |
|--------|---------|---------|-----|
| Gray Matter | 1200 | 80 | 0.85 |
| White Matter | 800 | 70 | 0.75 |
| CSF | 4000 | 2000 | 1.00 |
| Tumor | 1400 | 100 | 0.90 |
| Edema | 1500 | 120 | 0.92 |
| Necrosis | 2000 | 150 | 0.88 |
| Enhancement | 400 | 80 | 0.85 |

---

## Examples

### Example 1: Small, Slow-Growing Tumor

```bash
pft-simulate -o ./slow_tumor \
    --tumor-radius 3.0 \
    --proliferation-rate 0.005 \
    --diffusion-rate 0.05 \
    --duration 90
```

### Example 2: Aggressive, Fast-Growing Tumor

```bash
pft-simulate -o ./aggressive_tumor \
    --tumor-radius 5.0 \
    --proliferation-rate 0.025 \
    --diffusion-rate 0.3 \
    --duration 30
```

### Example 3: Off-Center Tumor Location

```bash
pft-simulate -o ./lateral_tumor \
    --tumor-center 15 -10 5 \
    --tumor-radius 4.0 \
    --duration 45
```

### Example 4: Python - Time Series Analysis

```python
from pft_fem import SUITAtlasLoader, MRISimulator, TumorParameters
import matplotlib.pyplot as plt

# Setup
loader = SUITAtlasLoader()
atlas_data = loader.load()
tumor_params = TumorParameters(center=(0, 0, 0), initial_radius=5.0)
simulator = MRISimulator(atlas_data, tumor_params)
simulator.setup()

# Collect time series data
volumes = []
displacements = []

def callback(state):
    vol = simulator.solver.compute_tumor_volume(state)
    disp = simulator.solver.compute_max_displacement(state)
    volumes.append((state.time, vol))
    displacements.append((state.time, disp))

states = simulator.simulate_growth(duration_days=60, callback=callback)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

times, vols = zip(*volumes)
ax1.plot(times, vols)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Tumor Volume (mm³)')
ax1.set_title('Tumor Growth Over Time')

times, disps = zip(*displacements)
ax2.plot(times, disps)
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Max Displacement (mm)')
ax2.set_title('Tissue Displacement Over Time')

plt.tight_layout()
plt.savefig('tumor_progression.png')
```

---

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pft_fem --cov-report=html

# Run specific test module
pytest tests/test_fem.py -v

# Run tests matching pattern
pytest -k "test_tumor" -v
```

### Test Structure

```
tests/
├── conftest.py      # Shared fixtures
├── test_atlas.py    # Atlas loading tests
├── test_mesh.py     # Mesh generation tests
└── test_fem.py      # FEM solver tests
```

---

## Project Structure

```
PFT_FEM/
├── pyproject.toml           # Project configuration
├── README.md                # This file
├── src/pft_fem/
│   ├── __init__.py         # Package exports
│   ├── cli.py              # Command-line interface
│   ├── atlas.py            # SUIT atlas loading
│   ├── mesh.py             # Mesh generation
│   ├── fem.py              # FEM solver
│   ├── simulation.py       # MRI simulation
│   └── io.py               # NIfTI I/O
└── tests/
    ├── conftest.py         # Test fixtures
    ├── test_atlas.py
    ├── test_mesh.py
    └── test_fem.py
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use PFT_FEM in your research, please cite:

```bibtex
@software{pft_fem,
  title = {PFT_FEM: Posterior Fossa Tumor Finite Element Modeling},
  year = {2024},
  url = {https://github.com/your-org/PFT_FEM}
}
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/PFT_FEM/issues)
- **Documentation**: This README
- **Examples**: See the [Examples](#examples) section
