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
from .fem import TumorGrowthSolver, TumorState, MaterialProperties, SolverConfig
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

    Defaults are configured for a non-infiltrative, expansile mass tumor
    (e.g., pilocytic astrocytoma or medulloblastoma) that grows as a solid
    mass filling ~30% of the posterior fossa with significant tissue displacement.

    Attributes:
        center: Tumor seed center in mm (relative to atlas origin).
        initial_radius: Initial tumor radius in mm.
        initial_density: Initial tumor cell density (0-1).
        proliferation_rate: Cell proliferation rate (1/day).
        diffusion_rate: Cell migration rate (mm^2/day) - low for non-infiltrative.
        necrotic_threshold: Density threshold for necrotic core.
        edema_extent: Extent of peritumoral edema in mm.
        enhancement_ring: Whether tumor has enhancing rim.
    """

    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_radius: float = 2.5  # Small seed for tumor growth
    initial_density: float = 0.9  # Higher density for solid tumor
    proliferation_rate: float = 0.04  # Higher rate for solid mass growth
    diffusion_rate: float = 0.01  # Very low - minimal infiltration
    necrotic_threshold: float = 0.99  # Very high - minimal central necrosis for uniform tumor
    edema_extent: float = 5.0  # Less edema for non-infiltrative tumor
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
    2. Generate FEM mesh (optionally coarse for speed)
    3. Simulate tumor growth
    4. Deform atlas based on simulation (interpolated to full resolution)
    5. Generate MRI signal based on tissue properties

    Supports multi-resolution simulation: compute on coarse mesh, output at
    full atlas resolution. Use SolverConfig.default() or SolverConfig.fast_coarse()
    for fast approximate solutions.

    When use_default_solver=True (the default), automatically loads the precomputed
    default solver for ~100x faster initialization.
    """

    def __init__(
        self,
        atlas_data: AtlasData,
        tumor_params: Optional[TumorParameters] = None,
        relaxation_params: Optional[Dict[str, TissueRelaxation]] = None,
        solver_config: Optional[SolverConfig] = None,
        use_default_solver: bool = True,
    ):
        """
        Initialize the MRI simulator.

        Args:
            atlas_data: Loaded SUIT atlas data.
            tumor_params: Tumor simulation parameters.
            relaxation_params: MRI relaxation parameters per tissue.
            solver_config: Solver configuration for performance/accuracy tradeoffs.
                          Defaults to SolverConfig.default() which uses coarse mesh
                          for fast simulation with high-resolution output.
            use_default_solver: If True (default), automatically load the precomputed
                               default solver when using default SolverConfig. This
                               provides ~100x faster initialization by skipping matrix
                               assembly. Set to False to always build solver from scratch.
        """
        self.atlas = atlas_data
        self.tumor_params = tumor_params or TumorParameters()
        self.relaxation_params = relaxation_params or DEFAULT_RELAXATION_PARAMS
        self.solver_config = solver_config or SolverConfig.default()
        self.use_default_solver = use_default_solver

        self.processor = AtlasProcessor(atlas_data)
        self.mesh: Optional[TetMesh] = None
        self.solver: Optional[TumorGrowthSolver] = None

        # Store original atlas properties for high-resolution output
        self._atlas_shape = atlas_data.shape
        self._atlas_affine = atlas_data.affine
        self._atlas_voxel_size = atlas_data.voxel_size

    def setup(self, mesh_resolution: Optional[float] = None) -> None:
        """
        Set up the simulation (mesh generation, solver initialization).

        Args:
            mesh_resolution: Target mesh voxel size in mm. If None, uses
                           solver_config.mesh_voxel_size (default: 3mm for speed).
        """
        # Try to load precomputed default solver if enabled
        if self.use_default_solver and mesh_resolution is None:
            if self._try_load_default_solver():
                return

        # Use solver_config mesh size if not explicitly specified
        if mesh_resolution is None:
            mesh_resolution = self.solver_config.mesh_voxel_size

        # Get tissue mask
        tissue_mask = self.processor.get_tissue_mask("all")

        # Downsample mask if using coarse mesh
        atlas_voxel_size = np.array(self._atlas_voxel_size)
        coarse_factor = mesh_resolution / atlas_voxel_size[0]  # Assume isotropic

        if coarse_factor > 1.5:
            # Downsample mask for coarse mesh generation
            from scipy import ndimage
            zoom_factors = tuple(1.0 / coarse_factor for _ in range(3))
            tissue_mask_coarse = ndimage.zoom(
                tissue_mask.astype(np.float32),
                zoom_factors,
                order=0  # Nearest neighbor for binary mask
            ) > 0.5

            # Use the original affine without modification
            # The mesh generator scales nodes by voxel_size first, then applies affine
            # This correctly maps coarse voxel corners to physical coordinates
            coarse_affine = self._atlas_affine.copy()

            coarse_voxel_size = tuple(v * coarse_factor for v in self._atlas_voxel_size)
        else:
            tissue_mask_coarse = tissue_mask
            coarse_affine = self._atlas_affine
            coarse_voxel_size = self._atlas_voxel_size

        # Generate mesh at specified resolution
        generator = MeshGenerator()
        self.mesh = generator.from_mask(
            mask=tissue_mask_coarse,
            voxel_size=coarse_voxel_size,
            labels=None,  # Labels not needed for coarse mesh
            affine=coarse_affine,
        )

        # Store mesh resolution info for later interpolation
        self._mesh_voxel_size = coarse_voxel_size
        self._mesh_affine = coarse_affine
        self._mesh_shape = tissue_mask_coarse.shape

        # Initialize FEM solver
        properties = self.tumor_params.to_material_properties()
        self.solver = TumorGrowthSolver(
            self.mesh,
            properties,
            solver_config=self.solver_config,
        )

    def _try_load_default_solver(self) -> bool:
        """
        Attempt to load the precomputed default solver.

        Returns:
            True if default solver was loaded successfully, False otherwise.
        """
        try:
            solver = TumorGrowthSolver.load_default(self.solver_config)
            mesh = solver.mesh

            # Check if mesh coordinates overlap with actual atlas tissue
            # The default solver is in MNI152 space; check tissue overlap with atlas
            inv_affine = np.linalg.inv(self._atlas_affine)
            nodes_on_tissue = 0
            sample_size = min(1000, len(mesh.nodes))  # Sample for efficiency

            for i in range(sample_size):
                node = mesh.nodes[i * len(mesh.nodes) // sample_size]
                node_h = np.append(node, 1.0)
                voxel = (inv_affine @ node_h)[:3].astype(int)
                if all(0 <= voxel[d] < self._atlas_shape[d] for d in range(3)):
                    if self.atlas.labels[voxel[0], voxel[1], voxel[2]] > 0:
                        nodes_on_tissue += 1

            # Require at least 30% of sampled mesh nodes to land on tissue
            tissue_coverage = nodes_on_tissue / sample_size
            if tissue_coverage < 0.3:
                # Insufficient tissue overlap - mesh and atlas have incompatible geometry
                return False

            self.solver = solver
            self.mesh = mesh

            # Set mesh resolution info from the loaded solver
            # The default solver uses 3mm voxel size
            self._mesh_voxel_size = (3.0, 3.0, 3.0)

            # Compute mesh affine by scaling the atlas affine
            # Only scale the diagonal (voxel size), not off-diagonal elements
            self._mesh_affine = self._atlas_affine.copy()
            scale_factor = 3.0 / np.array(self._atlas_voxel_size)
            for i in range(3):
                self._mesh_affine[:3, i] *= scale_factor[i]

            # Compute mesh shape from atlas shape at coarser resolution
            self._mesh_shape = tuple(
                int(s / (3.0 / v)) for s, v in zip(self._atlas_shape, self._atlas_voxel_size)
            )

            return True
        except FileNotFoundError:
            # Default solver not available, fall back to building from scratch
            return False
        except Exception:
            # Any other error, fall back to building from scratch
            return False

    def _create_initial_state(self) -> TumorState:
        """
        Create the initial tumor state based on tumor parameters.

        Returns:
            Initial TumorState with seed tumor.
        """
        if self.mesh is None:
            self.setup()

        seed_center = np.array(self.tumor_params.center)
        return TumorState.initial(
            mesh=self.mesh,
            seed_center=seed_center,
            seed_radius=self.tumor_params.initial_radius,
            seed_density=self.tumor_params.initial_density,
        )

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
        """
        Generate image for a single MRI sequence.

        Uses the SUIT template as the base intensity and modulates it by
        tissue-specific MRI signal equations. The template is deformed first
        to show tissue displacement from tumor mass effect.
        """
        shape = self.atlas.shape

        # Step 1: Apply deformation to the template to show tissue displacement
        deformed_template = self._apply_deformation(
            self.atlas.template.astype(np.float32),
            tumor_state
        )

        # Step 2: Apply deformation to labels for tissue segmentation
        deformed_labels = self._apply_deformation_labels(
            self.atlas.labels,
            tumor_state
        )

        # Step 3: Create tissue map from deformed labels (pass template for CSF detection)
        tissue_map = self._create_tissue_map_from_labels(
            deformed_labels, tumor_state, deformed_template
        )

        # Step 4: Compute tissue-specific signal modulation
        # Build modulation map based on tissue type
        modulation = np.ones(shape, dtype=np.float32)
        for tissue_name, mask in tissue_map.items():
            if tissue_name not in self.relaxation_params:
                continue

            params = self.relaxation_params[tissue_name]
            signal_factor = self._compute_signal(params, sequence, TR, TE, TI)
            # Normalize by a reference signal (gray matter)
            ref_params = self.relaxation_params["gray_matter"]
            ref_signal = self._compute_signal(ref_params, sequence, TR, TE, TI)
            if ref_signal > 0:
                modulation[mask] = signal_factor / ref_signal

        # Step 5: Generate MRI by modulating the deformed template
        # Create tissue mask (non-background regions)
        tissue_mask = deformed_template > 0

        # Normalize template to 0-1 range within tissue regions only
        if np.sum(tissue_mask) > 0:
            template_min = np.min(deformed_template[tissue_mask])
            template_max = np.max(deformed_template[tissue_mask])
            if template_max > template_min:
                normalized_template = np.zeros_like(deformed_template)
                normalized_template[tissue_mask] = (
                    deformed_template[tissue_mask] - template_min
                ) / (template_max - template_min)
                normalized_template = np.clip(normalized_template, 0, 1)
            else:
                normalized_template = deformed_template / (template_max + 1e-6)
        else:
            normalized_template = deformed_template / (np.max(deformed_template) + 1e-6)

        # Apply modulation to template
        image = normalized_template * modulation

        # Scale to typical MRI intensity range
        image = image * 1000

        # Add Rician noise (more realistic for magnitude MRI)
        noise_level = 0.02 * np.max(image)
        noise_real = np.random.normal(0, noise_level, shape).astype(np.float32)
        noise_imag = np.random.normal(0, noise_level, shape).astype(np.float32)
        image = np.sqrt((image + noise_real)**2 + noise_imag**2)

        return image.astype(np.float32)

    def _apply_deformation_labels(
        self,
        labels: NDArray[np.int32],
        tumor_state: TumorState,
    ) -> NDArray[np.int32]:
        """
        Apply tumor-induced deformation to label volume.

        Uses nearest-neighbor interpolation to preserve label values.
        """
        # Use same threshold as _apply_deformation
        if self.mesh is None or np.max(np.abs(tumor_state.displacement)) < 0.01:
            return labels

        from scipy import ndimage

        # Output at full atlas resolution
        output_shape = self._atlas_shape
        output_affine = self._atlas_affine
        output_voxel_size = self._atlas_voxel_size

        # Get mesh resolution info
        mesh_shape = getattr(self, '_mesh_shape', output_shape)
        mesh_affine = getattr(self, '_mesh_affine', output_affine)
        mesh_voxel_size = getattr(self, '_mesh_voxel_size', output_voxel_size)

        # Compute displacement field
        transform = compute_transform_from_simulation(
            displacement_at_nodes=tumor_state.displacement,
            mesh_nodes=self.mesh.nodes,
            volume_shape=mesh_shape,
            affine=mesh_affine,
            voxel_size=mesh_voxel_size,
            smoothing_sigma=2.0,
            output_shape=output_shape,
            output_affine=output_affine,
            output_voxel_size=output_voxel_size,
        )

        disp_field = transform.displacement_field

        # Create coordinate grids
        coords = np.array(np.meshgrid(
            np.arange(output_shape[0]),
            np.arange(output_shape[1]),
            np.arange(output_shape[2]),
            indexing='ij'
        ), dtype=np.float32)

        # Add displacement (convert mm to voxels)
        voxel_size = np.array(output_voxel_size)
        for d in range(3):
            coords[d] -= disp_field[..., d] / voxel_size[d]

        # Use nearest-neighbor interpolation for labels
        deformed = ndimage.map_coordinates(
            labels.astype(np.float32), coords, order=0, mode='constant', cval=0
        )

        return deformed.astype(np.int32)

    def _create_tissue_map_from_labels(
        self,
        labels: NDArray[np.int32],
        tumor_state: TumorState,
        deformed_template: Optional[NDArray[np.float32]] = None,
    ) -> Dict[str, NDArray[np.bool_]]:
        """
        Create tissue segmentation from (possibly deformed) labels.

        SUIT atlas labels:
        - Labels 1-28: Cerebellar lobules (cortical gray matter)
        - Labels 29-34: Deep cerebellar nuclei (gray matter structures)

        For the synthetic atlas:
        - Labels 5-7: Cerebellum (gray matter)
        - Label 29: Brainstem (treated as white matter)
        - Label 30: Fourth ventricle (CSF)

        We detect which atlas is being used based on the label values present.
        """
        tissue_map = {}

        # Detect atlas type based on labels present
        unique_labels = np.unique(labels)
        has_nuclei_labels = any(l in unique_labels for l in [31, 32, 33, 34])

        if has_nuclei_labels:
            # Real SUIT atlas: labels 1-28 = lobules, 29-34 = nuclei (all gray matter)
            # No explicit white matter or CSF labels
            tissue_map["gray_matter"] = (labels >= 1) & (labels <= 34)
            tissue_map["white_matter"] = np.zeros_like(labels, dtype=bool)
            tissue_map["csf"] = np.zeros_like(labels, dtype=bool)

            # Use template intensity to identify CSF (low intensity regions)
            if deformed_template is not None:
                template_mask = labels > 0
                if np.sum(template_mask) > 0:
                    # CSF typically has very low T1 signal
                    threshold = np.percentile(deformed_template[template_mask], 10)
                    tissue_map["csf"] = (deformed_template < threshold) & template_mask
                    tissue_map["gray_matter"] = tissue_map["gray_matter"] & ~tissue_map["csf"]
        else:
            # Synthetic atlas or atlas with explicit tissue labels
            # Labels 1-28: gray matter (cerebellar lobules)
            tissue_map["gray_matter"] = (labels >= 1) & (labels <= 28)
            # Label 29: brainstem/white matter (in synthetic atlas)
            tissue_map["white_matter"] = labels == 29
            # Label 30: fourth ventricle/CSF (in synthetic atlas)
            tissue_map["csf"] = labels == 30

        # Map tumor density to voxels (already interpolated from mesh nodes)
        tumor_density = self._interpolate_to_volume(tumor_state.cell_density)

        # NOTE: Do NOT deform tumor_density here - it's already in deformed space
        # since it was interpolated from deformed mesh node positions

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
            if tissue in tissue_map:
                tissue_map[tissue] = tissue_map[tissue] & ~tumor_core & ~edema

        return tissue_map

    def _create_tissue_map(
        self,
        tumor_state: TumorState,
    ) -> Dict[str, NDArray[np.bool_]]:
        """
        Create tissue segmentation including tumor.

        DEPRECATED: Use _create_tissue_map_from_labels with deformed labels instead.
        This method is kept for backward compatibility.
        """
        # Delegate to the new method with undeformed labels and template
        return self._create_tissue_map_from_labels(
            self.atlas.labels,
            tumor_state,
            self.atlas.template.astype(np.float32),
        )

    def _interpolate_to_volume(
        self,
        node_values: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """
        Interpolate node values to volume grid using FEM shape functions.

        Uses proper tetrahedral interpolation to create a continuous field
        rather than discrete point samples. Each voxel's value is computed
        by finding which element contains it and using barycentric interpolation.
        """
        shape = self.atlas.shape
        volume = np.zeros(shape, dtype=np.float32)

        if self.mesh is None:
            return volume

        from scipy import ndimage

        # Precompute inverse affine for coordinate transform
        inv_affine = np.linalg.inv(self.atlas.affine)

        # Get elements with significant values (optimization)
        elem_max_values = np.array([
            np.max(node_values[self.mesh.elements[e]])
            for e in range(self.mesh.num_elements)
        ])
        active_elements = np.where(elem_max_values > 1e-6)[0]

        # Process each active element
        for e in active_elements:
            elem_nodes = self.mesh.elements[e]
            elem_coords = self.mesh.nodes[elem_nodes]  # (4, 3) physical coords
            elem_values = node_values[elem_nodes]  # (4,) values

            # Convert element nodes to voxel coordinates
            elem_voxels = np.zeros((4, 3))
            for i in range(4):
                homogeneous = np.append(elem_coords[i], 1.0)
                elem_voxels[i] = (inv_affine @ homogeneous)[:3]

            # Get bounding box in voxel space (with 1-voxel padding)
            vmin = np.floor(elem_voxels.min(axis=0) - 1).astype(int)
            vmax = np.ceil(elem_voxels.max(axis=0) + 1).astype(int)

            # Clip to volume bounds
            vmin = np.maximum(vmin, 0)
            vmax = np.minimum(vmax, np.array(shape))

            # Precompute barycentric transform matrix for this tetrahedron
            # Using physical coordinates for accuracy
            T = np.column_stack([
                elem_coords[0] - elem_coords[3],
                elem_coords[1] - elem_coords[3],
                elem_coords[2] - elem_coords[3],
            ])

            # Check if tetrahedron is degenerate
            det = np.linalg.det(T)
            if abs(det) < 1e-12:
                continue

            T_inv = np.linalg.inv(T)

            # Sample all voxels in bounding box
            for vx in range(vmin[0], vmax[0]):
                for vy in range(vmin[1], vmax[1]):
                    for vz in range(vmin[2], vmax[2]):
                        # Convert voxel center to physical coordinates
                        voxel_center = np.array([vx + 0.5, vy + 0.5, vz + 0.5, 1.0])
                        phys_coord = (self.atlas.affine @ voxel_center)[:3]

                        # Compute barycentric coordinates
                        delta = phys_coord - elem_coords[3]
                        bary = T_inv @ delta
                        bary4 = np.array([bary[0], bary[1], bary[2], 1 - bary.sum()])

                        # Check if point is inside tetrahedron (all bary coords in [0,1])
                        if np.all(bary4 >= -0.01) and np.all(bary4 <= 1.01):
                            # Interpolate value using shape functions
                            interp_value = np.dot(bary4, elem_values)

                            # Use maximum value for overlapping elements
                            if interp_value > volume[vx, vy, vz]:
                                volume[vx, vy, vz] = interp_value

        # Light smoothing to reduce any remaining blockiness
        volume = ndimage.gaussian_filter(volume, sigma=0.3)

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
        """
        Apply tumor-induced deformation to image.

        Uses multi-resolution interpolation: computes displacement from coarse
        mesh nodes and interpolates to full atlas resolution.
        """
        # Lower threshold to ensure small displacements are visible
        if self.mesh is None or np.max(np.abs(tumor_state.displacement)) < 0.01:
            return image

        from scipy import ndimage

        # Output at full atlas resolution
        output_shape = self._atlas_shape
        output_affine = self._atlas_affine
        output_voxel_size = self._atlas_voxel_size

        # Get mesh resolution info
        mesh_shape = getattr(self, '_mesh_shape', output_shape)
        mesh_affine = getattr(self, '_mesh_affine', output_affine)
        mesh_voxel_size = getattr(self, '_mesh_voxel_size', output_voxel_size)

        # Compute displacement field at full resolution using multi-resolution interpolation
        transform = compute_transform_from_simulation(
            displacement_at_nodes=tumor_state.displacement,
            mesh_nodes=self.mesh.nodes,
            volume_shape=mesh_shape,
            affine=mesh_affine,
            voxel_size=mesh_voxel_size,
            smoothing_sigma=2.0,
            output_shape=output_shape,
            output_affine=output_affine,
            output_voxel_size=output_voxel_size,
        )

        disp_field = transform.displacement_field

        # Apply deformation using coordinate transform
        # Create coordinate grids at full resolution
        coords = np.array(np.meshgrid(
            np.arange(output_shape[0]),
            np.arange(output_shape[1]),
            np.arange(output_shape[2]),
            indexing='ij'
        ), dtype=np.float32)

        # Add displacement (convert mm to voxels)
        voxel_size = np.array(output_voxel_size)
        for d in range(3):
            coords[d] -= disp_field[..., d] / voxel_size[d]

        # Interpolate image at new coordinates
        deformed = ndimage.map_coordinates(
            image, coords, order=1, mode='constant', cval=0
        )

        return deformed.astype(np.float32)

    def _apply_deformation_antspy(
        self,
        image: NDArray[np.float32],
        tumor_state: TumorState,
        interpolation: str = "linear",
    ) -> NDArray[np.float32]:
        """
        Apply tumor-induced deformation to image using ANTsPy.

        Uses ANTsPy's apply_transforms for proper image warping with
        a displacement field stored as a NIfTI vector image. This provides
        more accurate interpolation than scipy's map_coordinates for
        large deformations.

        Args:
            image: Input image to deform.
            tumor_state: Current tumor state with displacement field.
            interpolation: Interpolation method ("linear", "nearestNeighbor",
                          "bSpline", "genericLabel").

        Returns:
            Deformed image.

        Raises:
            ImportError: If ANTsPy is not installed.
        """
        try:
            import ants
        except ImportError:
            raise ImportError(
                "ANTsPy is required for _apply_deformation_antspy. "
                "Install with: pip install antspyx"
            )

        # Use same threshold as _apply_deformation
        if self.mesh is None or np.max(np.abs(tumor_state.displacement)) < 0.01:
            return image

        # Output at full atlas resolution
        output_shape = self._atlas_shape
        output_affine = self._atlas_affine
        output_voxel_size = self._atlas_voxel_size

        # Get mesh resolution info
        mesh_shape = getattr(self, '_mesh_shape', output_shape)
        mesh_affine = getattr(self, '_mesh_affine', output_affine)
        mesh_voxel_size = getattr(self, '_mesh_voxel_size', output_voxel_size)

        # Compute displacement field from FEM nodes
        transform = compute_transform_from_simulation(
            displacement_at_nodes=tumor_state.displacement,
            mesh_nodes=self.mesh.nodes,
            volume_shape=mesh_shape,
            affine=mesh_affine,
            voxel_size=mesh_voxel_size,
            smoothing_sigma=2.0,
            output_shape=output_shape,
            output_affine=output_affine,
            output_voxel_size=output_voxel_size,
        )

        disp_field = transform.displacement_field

        # Convert displacement field to ANTsPy format
        # ANTsPy expects (X, Y, Z, 1, 3) with vector components as last dimension
        # and values in physical coordinates (mm)
        disp_ants_data = disp_field[:, :, :, np.newaxis, :].astype(np.float32)

        # Create ANTsPy image for displacement field
        # ANTsPy uses ITK convention: displacement is from fixed to moving
        # Our displacement is from original to deformed, so we need to negate
        # for pullback (sampling deformed positions in original image)
        disp_ants_data = -disp_ants_data

        # Create ANTs displacement field image
        # Set up header with correct dimensions and directions
        import nibabel as nib
        import tempfile
        import os

        # Create temporary files for ANTsPy I/O
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save displacement field as NIfTI
            disp_nifti = nib.Nifti1Image(disp_ants_data, output_affine)
            disp_nifti.header.set_intent("vector", name="displacement")
            disp_nifti.header.set_data_dtype(np.float32)
            disp_path = os.path.join(tmpdir, "disp.nii.gz")
            nib.save(disp_nifti, disp_path)

            # Save input image as NIfTI
            img_nifti = nib.Nifti1Image(image, output_affine)
            img_path = os.path.join(tmpdir, "input.nii.gz")
            nib.save(img_nifti, img_path)

            # Load with ANTsPy
            ants_image = ants.image_read(img_path)
            ants_warp = ants.image_read(disp_path)

            # Apply transformation
            # The displacement field is applied as a deformation field
            warped = ants.apply_transforms(
                fixed=ants_image,
                moving=ants_image,
                transformlist=[disp_path],
                interpolator=interpolation,
                whichtoinvert=[False],
            )

            # Convert back to numpy array
            result = warped.numpy()

        return result.astype(np.float32)

    def generate_mri_with_antspy_warping(
        self,
        tumor_state: TumorState,
        sequences: Optional[List[MRISequence]] = None,
        TR: float = 500.0,
        TE: float = 15.0,
        TI: float = 1200.0,
    ) -> Dict[str, NDArray[np.float32]]:
        """
        Generate synthetic MRI images using ANTsPy for deformation.

        This method uses ANTsPy's apply_transforms for more accurate
        image warping, particularly for large deformations.

        Args:
            tumor_state: Current tumor state from simulation.
            sequences: List of MRI sequences to generate.
            TR: Repetition time in ms.
            TE: Echo time in ms.
            TI: Inversion time in ms (for FLAIR).

        Returns:
            Dictionary mapping sequence name to image volume.

        Raises:
            ImportError: If ANTsPy is not installed.
        """
        if sequences is None:
            sequences = [MRISequence.T1, MRISequence.T2, MRISequence.FLAIR]

        images = {}

        for seq in sequences:
            images[seq.value] = self._generate_sequence_antspy(
                tumor_state, seq, TR, TE, TI
            )

        return images

    def _generate_sequence_antspy(
        self,
        tumor_state: TumorState,
        sequence: MRISequence,
        TR: float,
        TE: float,
        TI: float,
    ) -> NDArray[np.float32]:
        """
        Generate image for a single MRI sequence using ANTsPy warping.

        Similar to _generate_sequence but uses ANTsPy for deformation.
        """
        shape = self.atlas.shape

        # Step 1: Apply deformation to the template using ANTsPy
        deformed_template = self._apply_deformation_antspy(
            self.atlas.template.astype(np.float32),
            tumor_state,
            interpolation="linear"
        )

        # Step 2: Apply deformation to labels using nearest-neighbor
        deformed_labels = self._apply_deformation_antspy(
            self.atlas.labels.astype(np.float32),
            tumor_state,
            interpolation="nearestNeighbor"
        ).astype(np.int32)

        # Step 3: Create tissue map from deformed labels (pass template for CSF detection)
        tissue_map = self._create_tissue_map_from_labels(
            deformed_labels, tumor_state, deformed_template
        )

        # Step 4: Compute tissue-specific signal modulation
        modulation = np.ones(shape, dtype=np.float32)
        for tissue_name, mask in tissue_map.items():
            if tissue_name not in self.relaxation_params:
                continue

            params = self.relaxation_params[tissue_name]
            signal_factor = self._compute_signal(params, sequence, TR, TE, TI)
            ref_params = self.relaxation_params["gray_matter"]
            ref_signal = self._compute_signal(ref_params, sequence, TR, TE, TI)
            if ref_signal > 0:
                modulation[mask] = signal_factor / ref_signal

        # Step 5: Generate MRI by modulating the deformed template
        # Create tissue mask (non-background regions)
        tissue_mask = deformed_template > 0

        # Normalize template to 0-1 range within tissue regions only
        if np.sum(tissue_mask) > 0:
            template_min = np.min(deformed_template[tissue_mask])
            template_max = np.max(deformed_template[tissue_mask])
            if template_max > template_min:
                normalized_template = np.zeros_like(deformed_template)
                normalized_template[tissue_mask] = (
                    deformed_template[tissue_mask] - template_min
                ) / (template_max - template_min)
                normalized_template = np.clip(normalized_template, 0, 1)
            else:
                normalized_template = deformed_template / (template_max + 1e-6)
        else:
            normalized_template = deformed_template / (np.max(deformed_template) + 1e-6)

        image = normalized_template * modulation * 1000

        # Add Rician noise
        noise_level = 0.02 * np.max(image)
        noise_real = np.random.normal(0, noise_level, shape).astype(np.float32)
        noise_imag = np.random.normal(0, noise_level, shape).astype(np.float32)
        image = np.sqrt((image + noise_real)**2 + noise_imag**2)

        return image.astype(np.float32)

    def _compute_spatial_transform(
        self,
        tumor_state: TumorState,
    ) -> Optional[SpatialTransform]:
        """
        Compute the complete spatial transform from SUIT template to deformed state.

        This transform encapsulates the deformation induced by tumor growth and
        can be exported in ANTsPy-compatible formats for use in other pipelines.

        Uses multi-resolution interpolation: computes displacement from coarse
        mesh nodes and outputs at full atlas resolution.

        Args:
            tumor_state: Final tumor state with displacement field.

        Returns:
            SpatialTransform instance, or None if no mesh is available.
        """
        if self.mesh is None:
            return None

        # Skip if displacement is negligible
        if np.max(np.abs(tumor_state.displacement)) < 0.01:
            # Return identity transform at full atlas resolution
            return SpatialTransform.identity(
                shape=self._atlas_shape,
                voxel_size=self._atlas_voxel_size,
                affine=self._atlas_affine,
            )

        # Get mesh resolution info
        mesh_shape = getattr(self, '_mesh_shape', self._atlas_shape)
        mesh_affine = getattr(self, '_mesh_affine', self._atlas_affine)
        mesh_voxel_size = getattr(self, '_mesh_voxel_size', self._atlas_voxel_size)

        # Compute transform from FEM node displacements with multi-resolution output
        transform = compute_transform_from_simulation(
            displacement_at_nodes=tumor_state.displacement,
            mesh_nodes=self.mesh.nodes,
            volume_shape=mesh_shape,
            affine=mesh_affine,
            voxel_size=mesh_voxel_size,
            smoothing_sigma=2.0,
            # Output at full atlas resolution
            output_shape=self._atlas_shape,
            output_affine=self._atlas_affine,
            output_voxel_size=self._atlas_voxel_size,
        )

        # Add simulation metadata to transform
        transform.metadata = {
            "simulation_time_days": tumor_state.time,
            "tumor_center": self.tumor_params.center,
            "tumor_initial_radius": self.tumor_params.initial_radius,
            "proliferation_rate": self.tumor_params.proliferation_rate,
            "diffusion_rate": self.tumor_params.diffusion_rate,
            "mesh_voxel_size": self.solver_config.mesh_voxel_size,
            "output_at_full_resolution": self.solver_config.output_at_full_resolution,
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
