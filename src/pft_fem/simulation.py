"""
MRI Simulation module for generating synthetic images.

Combines atlas data, tumor growth simulation, and MRI signal modeling
to produce realistic synthetic NIfTI images.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .atlas import AtlasData, AtlasProcessor
from .mesh import TetMesh, MeshGenerator
from .fem import TumorGrowthSolver, TumorState, MaterialProperties
from .transforms import SpatialTransform, compute_transform_from_simulation


class MRISequence(Enum):
    """Common MRI pulse sequences."""

    T1 = "T1"
    T2 = "T2"
    FLAIR = "FLAIR"
    T1_CONTRAST = "T1_contrast"
    DWI = "DWI"


@dataclass
class TissueRelaxation:
    """MRI relaxation parameters for a tissue type."""

    T1: float  # T1 relaxation time in ms
    T2: float  # T2 relaxation time in ms
    PD: float  # Proton density (0-1)


# Literature-based relaxation times at 1.5T
DEFAULT_RELAXATION_PARAMS = {
    "gray_matter": TissueRelaxation(T1=1200, T2=80, PD=0.85),
    "white_matter": TissueRelaxation(T1=800, T2=70, PD=0.75),
    "csf": TissueRelaxation(T1=4000, T2=2000, PD=1.0),
    "tumor": TissueRelaxation(T1=1400, T2=100, PD=0.90),
    "edema": TissueRelaxation(T1=1500, T2=120, PD=0.92),
    "necrosis": TissueRelaxation(T1=2000, T2=150, PD=0.88),
    "enhancement": TissueRelaxation(T1=400, T2=80, PD=0.85),  # After contrast
}


@dataclass
class TumorParameters:
    """
    Parameters defining tumor characteristics for simulation.

    Attributes:
        center: Tumor seed center in mm (relative to atlas origin).
        initial_radius: Initial tumor radius in mm.
        initial_density: Initial tumor cell density (0-1).
        proliferation_rate: Cell proliferation rate (1/day).
        diffusion_rate: Cell migration rate (mm^2/day).
        necrotic_threshold: Density threshold for necrotic core.
        edema_extent: Extent of peritumoral edema in mm.
        enhancement_ring: Whether tumor has enhancing rim.
    """

    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_radius: float = 5.0
    initial_density: float = 0.8
    proliferation_rate: float = 0.012
    diffusion_rate: float = 0.15
    necrotic_threshold: float = 0.9
    edema_extent: float = 10.0
    enhancement_ring: bool = True

    def to_material_properties(self) -> MaterialProperties:
        """Convert to FEM material properties."""
        return MaterialProperties(
            proliferation_rate=self.proliferation_rate,
            diffusion_coefficient=self.diffusion_rate,
        )


@dataclass
class SimulationResult:
    """
    Container for simulation results.

    Attributes:
        tumor_states: Tumor state at each time point.
        mri_images: Dictionary of MRI sequence -> volume.
        deformed_atlas: Atlas with tumor-induced deformation.
        tumor_mask: Binary tumor mask.
        edema_mask: Binary edema mask.
        spatial_transform: Complete spatial transform from SUIT to deformed space.
        metadata: Additional simulation metadata.
    """

    tumor_states: List[TumorState]
    mri_images: Dict[str, NDArray[np.float32]]
    deformed_atlas: NDArray[np.float32]
    tumor_mask: NDArray[np.bool_]
    edema_mask: NDArray[np.bool_]
    spatial_transform: Optional[SpatialTransform] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MRISimulator:
    """
    Simulator for generating synthetic MRI images with tumors.

    Pipeline:
    1. Load atlas data
    2. Generate FEM mesh
    3. Simulate tumor growth
    4. Deform atlas based on simulation
    5. Generate MRI signal based on tissue properties
    """

    def __init__(
        self,
        atlas_data: AtlasData,
        tumor_params: Optional[TumorParameters] = None,
        relaxation_params: Optional[Dict[str, TissueRelaxation]] = None,
    ):
        """
        Initialize the MRI simulator.

        Args:
            atlas_data: Loaded SUIT atlas data.
            tumor_params: Tumor simulation parameters.
            relaxation_params: MRI relaxation parameters per tissue.
        """
        self.atlas = atlas_data
        self.tumor_params = tumor_params or TumorParameters()
        self.relaxation_params = relaxation_params or DEFAULT_RELAXATION_PARAMS

        self.processor = AtlasProcessor(atlas_data)
        self.mesh: Optional[TetMesh] = None
        self.solver: Optional[TumorGrowthSolver] = None

    def setup(self, mesh_resolution: float = 2.0) -> None:
        """
        Set up the simulation (mesh generation, solver initialization).

        Args:
            mesh_resolution: Target mesh element size in mm.
        """
        # Get tissue mask
        tissue_mask = self.processor.get_tissue_mask("all")

        # Generate mesh
        generator = MeshGenerator()
        self.mesh = generator.from_mask(
            mask=tissue_mask,
            voxel_size=self.atlas.voxel_size,
            labels=self.atlas.labels,
            affine=self.atlas.affine,
        )

        # Initialize FEM solver
        properties = self.tumor_params.to_material_properties()
        self.solver = TumorGrowthSolver(self.mesh, properties)

    def simulate_growth(
        self,
        duration_days: float = 30.0,
        time_step: float = 1.0,
        verbose: bool = False,
    ) -> List[TumorState]:
        """
        Run tumor growth simulation.

        Args:
            duration_days: Simulation duration in days.
            time_step: Time step in days.
            verbose: Whether to print progress.

        Returns:
            List of TumorState at each time step.
        """
        if self.mesh is None or self.solver is None:
            self.setup()

        # Create initial state
        seed_center = np.array(self.tumor_params.center)
        initial_state = TumorState.initial(
            mesh=self.mesh,
            seed_center=seed_center,
            seed_radius=self.tumor_params.initial_radius,
            seed_density=self.tumor_params.initial_density,
        )

        # Run simulation
        def callback(state, step):
            if verbose and step % 10 == 0:
                vol = self.solver.compute_tumor_volume(state)
                print(f"Day {state.time:.1f}: Tumor volume = {vol:.2f} mm³")

        states = self.solver.simulate(
            initial_state=initial_state,
            duration=duration_days,
            dt=time_step,
            callback=callback if verbose else None,
        )

        return states

    def generate_mri(
        self,
        tumor_state: TumorState,
        sequences: Optional[List[MRISequence]] = None,
        TR: float = 500.0,
        TE: float = 15.0,
        TI: float = 1200.0,
    ) -> Dict[str, NDArray[np.float32]]:
        """
        Generate synthetic MRI images for specified sequences.

        Args:
            tumor_state: Current tumor state from simulation.
            sequences: List of MRI sequences to generate.
            TR: Repetition time in ms.
            TE: Echo time in ms.
            TI: Inversion time in ms (for FLAIR).

        Returns:
            Dictionary mapping sequence name to image volume.
        """
        if sequences is None:
            sequences = [MRISequence.T1, MRISequence.T2, MRISequence.FLAIR]

        images = {}

        for seq in sequences:
            images[seq.value] = self._generate_sequence(
                tumor_state, seq, TR, TE, TI
            )

        return images

    def _generate_sequence(
        self,
        tumor_state: TumorState,
        sequence: MRISequence,
        TR: float,
        TE: float,
        TI: float,
    ) -> NDArray[np.float32]:
        """Generate image for a single MRI sequence."""
        shape = self.atlas.shape
        image = np.zeros(shape, dtype=np.float32)

        # Get tissue segmentation with tumor
        tissue_map = self._create_tissue_map(tumor_state)

        # Compute signal intensity for each tissue
        for tissue_name, mask in tissue_map.items():
            if tissue_name not in self.relaxation_params:
                continue

            params = self.relaxation_params[tissue_name]
            signal = self._compute_signal(params, sequence, TR, TE, TI)
            image[mask] = signal

        # Add noise
        noise_level = 0.02 * np.max(image)
        noise = np.random.normal(0, noise_level, shape).astype(np.float32)
        image = np.maximum(image + noise, 0)

        # Apply deformation from tumor mass effect
        image = self._apply_deformation(image, tumor_state)

        return image

    def _create_tissue_map(
        self,
        tumor_state: TumorState,
    ) -> Dict[str, NDArray[np.bool_]]:
        """Create tissue segmentation including tumor."""
        tissue_map = {}

        # Base tissues from atlas
        labels = self.atlas.labels

        # Gray matter (cerebellar cortex)
        tissue_map["gray_matter"] = (labels >= 1) & (labels <= 28)

        # CSF (fourth ventricle)
        tissue_map["csf"] = labels == 30

        # White matter (approximate as brainstem)
        tissue_map["white_matter"] = labels == 29

        # Map tumor density to voxels
        tumor_density = self._interpolate_to_volume(
            tumor_state.cell_density
        )

        # Define tumor regions based on density
        tumor_core = tumor_density > 0.5
        tumor_rim = (tumor_density > 0.1) & (tumor_density <= 0.5)

        # Necrotic core (very high density region)
        necrotic = tumor_density > self.tumor_params.necrotic_threshold

        # Edema (around tumor)
        from scipy import ndimage
        dilated = ndimage.binary_dilation(
            tumor_density > 0.1,
            iterations=int(self.tumor_params.edema_extent / 2)
        )
        edema = dilated & (tumor_density < 0.1) & (labels > 0)

        # Override base tissues with tumor/edema
        tissue_map["tumor"] = tumor_core & ~necrotic
        tissue_map["necrosis"] = necrotic
        tissue_map["edema"] = edema

        if self.tumor_params.enhancement_ring:
            tissue_map["enhancement"] = tumor_rim

        # Remove tumor regions from normal tissue
        for tissue in ["gray_matter", "white_matter"]:
            tissue_map[tissue] = tissue_map[tissue] & ~tumor_core & ~edema

        return tissue_map

    def _interpolate_to_volume(
        self,
        node_values: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """Interpolate node values to volume grid."""
        shape = self.atlas.shape
        volume = np.zeros(shape, dtype=np.float32)

        if self.mesh is None:
            return volume

        # Simple nearest-neighbor interpolation
        # For better results, use proper FEM interpolation
        for i, node in enumerate(self.mesh.nodes):
            # Convert physical coords to voxel coords
            voxel = self._physical_to_voxel(node)

            # Check bounds
            if all(0 <= voxel[d] < shape[d] for d in range(3)):
                vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
                volume[vx, vy, vz] = max(volume[vx, vy, vz], node_values[i])

        # Smooth to fill gaps
        from scipy import ndimage
        volume = ndimage.gaussian_filter(volume, sigma=1.0)

        return volume

    def _physical_to_voxel(
        self,
        physical_coords: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert physical coordinates to voxel indices."""
        # Apply inverse affine
        inv_affine = np.linalg.inv(self.atlas.affine)
        homogeneous = np.append(physical_coords, 1.0)
        voxel = (inv_affine @ homogeneous)[:3]
        return voxel

    def _compute_signal(
        self,
        params: TissueRelaxation,
        sequence: MRISequence,
        TR: float,
        TE: float,
        TI: float,
    ) -> float:
        """Compute MRI signal intensity for a tissue and sequence."""
        T1, T2, PD = params.T1, params.T2, params.PD

        if sequence == MRISequence.T1:
            # Spin echo T1-weighted
            signal = PD * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)

        elif sequence == MRISequence.T2:
            # Spin echo T2-weighted
            signal = PD * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)
            # T2 weighting emphasized by longer TE
            signal *= np.exp(-TE / T2)

        elif sequence == MRISequence.FLAIR:
            # Fluid-attenuated inversion recovery
            # Suppresses CSF signal
            signal = PD * np.abs(
                1 - 2 * np.exp(-TI / T1) + np.exp(-TR / T1)
            ) * np.exp(-TE / T2)

        elif sequence == MRISequence.T1_CONTRAST:
            # T1 with gadolinium contrast
            # Contrast shortens T1 in enhancing regions
            effective_T1 = T1 * 0.3 if params == self.relaxation_params.get("enhancement") else T1
            signal = PD * (1 - np.exp(-TR / effective_T1)) * np.exp(-TE / T2)

        elif sequence == MRISequence.DWI:
            # Diffusion-weighted imaging (simplified)
            b_value = 1000  # s/mm^2
            ADC = 0.8e-3  # mm^2/s for normal tissue
            if params == self.relaxation_params.get("tumor"):
                ADC = 0.5e-3  # Reduced in tumor
            signal = PD * np.exp(-b_value * ADC) * np.exp(-TE / T2)

        else:
            signal = PD

        return float(signal * 1000)  # Scale to typical MRI range

    def _apply_deformation(
        self,
        image: NDArray[np.float32],
        tumor_state: TumorState,
    ) -> NDArray[np.float32]:
        """Apply tumor-induced deformation to image."""
        if self.mesh is None or np.max(np.abs(tumor_state.displacement)) < 0.1:
            return image

        from scipy import ndimage

        # Create displacement field in volume space
        shape = self.atlas.shape
        disp_field = np.zeros((*shape, 3), dtype=np.float32)

        # Interpolate displacement from nodes to volume
        for i, node in enumerate(self.mesh.nodes):
            voxel = self._physical_to_voxel(node)

            if all(0 <= voxel[d] < shape[d] for d in range(3)):
                vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
                disp_field[vx, vy, vz] = tumor_state.displacement[i]

        # Smooth displacement field
        for d in range(3):
            disp_field[..., d] = ndimage.gaussian_filter(
                disp_field[..., d], sigma=2.0
            )

        # Apply deformation using coordinate transform
        # Create coordinate grids
        coords = np.array(np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        ), dtype=np.float32)

        # Add displacement (convert mm to voxels)
        voxel_size = np.array(self.atlas.voxel_size)
        for d in range(3):
            coords[d] -= disp_field[..., d] / voxel_size[d]

        # Interpolate image at new coordinates
        deformed = ndimage.map_coordinates(
            image, coords, order=1, mode='constant', cval=0
        )

        return deformed.astype(np.float32)

    def _compute_spatial_transform(
        self,
        tumor_state: TumorState,
    ) -> Optional[SpatialTransform]:
        """
        Compute the complete spatial transform from SUIT template to deformed state.

        This transform encapsulates the deformation induced by tumor growth and
        can be exported in ANTsPy-compatible formats for use in other pipelines.

        Args:
            tumor_state: Final tumor state with displacement field.

        Returns:
            SpatialTransform instance, or None if no mesh is available.
        """
        if self.mesh is None:
            return None

        # Skip if displacement is negligible
        if np.max(np.abs(tumor_state.displacement)) < 0.01:
            # Return identity transform
            return SpatialTransform.identity(
                shape=self.atlas.shape,
                voxel_size=self.atlas.voxel_size,
                affine=self.atlas.affine,
            )

        # Compute transform from FEM node displacements
        transform = compute_transform_from_simulation(
            displacement_at_nodes=tumor_state.displacement,
            mesh_nodes=self.mesh.nodes,
            volume_shape=self.atlas.shape,
            affine=self.atlas.affine,
            voxel_size=self.atlas.voxel_size,
            smoothing_sigma=2.0,
        )

        # Add simulation metadata to transform
        transform.metadata = {
            "simulation_time_days": tumor_state.time,
            "tumor_center": self.tumor_params.center,
            "tumor_initial_radius": self.tumor_params.initial_radius,
            "proliferation_rate": self.tumor_params.proliferation_rate,
            "diffusion_rate": self.tumor_params.diffusion_rate,
        }

        return transform

    def run_full_pipeline(
        self,
        duration_days: float = 30.0,
        sequences: Optional[List[MRISequence]] = None,
        verbose: bool = False,
    ) -> SimulationResult:
        """
        Run complete simulation pipeline.

        Args:
            duration_days: Simulation duration.
            sequences: MRI sequences to generate.
            verbose: Print progress information.

        Returns:
            SimulationResult with all outputs.
        """
        if verbose:
            print("Setting up simulation...")
        self.setup()

        if verbose:
            print("Running tumor growth simulation...")
        states = self.simulate_growth(duration_days, verbose=verbose)

        final_state = states[-1]

        if verbose:
            print("Generating MRI images...")
        mri_images = self.generate_mri(final_state, sequences)

        # Create masks
        tumor_density = self._interpolate_to_volume(final_state.cell_density)
        tumor_mask = tumor_density > 0.1

        from scipy import ndimage
        dilated = ndimage.binary_dilation(tumor_mask, iterations=5)
        edema_mask = dilated & ~tumor_mask & (self.atlas.labels > 0)

        # Deformed atlas
        deformed_atlas = self._apply_deformation(
            self.atlas.template.copy(),
            final_state
        )

        # Compute spatial transform from SUIT template to deformed state
        if verbose:
            print("Computing spatial transform...")
        spatial_transform = self._compute_spatial_transform(final_state)

        # Compute metadata
        tumor_volume = self.solver.compute_tumor_volume(final_state)
        max_displacement = self.solver.compute_max_displacement(final_state)

        metadata = {
            "duration_days": duration_days,
            "final_tumor_volume_mm3": tumor_volume,
            "max_displacement_mm": max_displacement,
            "tumor_params": {
                "center": self.tumor_params.center,
                "initial_radius": self.tumor_params.initial_radius,
                "proliferation_rate": self.tumor_params.proliferation_rate,
                "diffusion_rate": self.tumor_params.diffusion_rate,
            },
            "atlas_shape": self.atlas.shape,
            "voxel_size": self.atlas.voxel_size,
            "spatial_transform_info": spatial_transform.to_dict() if spatial_transform else None,
        }

        return SimulationResult(
            tumor_states=states,
            mri_images=mri_images,
            deformed_atlas=deformed_atlas,
            tumor_mask=tumor_mask,
            edema_mask=edema_mask,
            spatial_transform=spatial_transform,
            metadata=metadata,
        )
