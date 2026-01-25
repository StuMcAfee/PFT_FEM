"""
Command-line interface for PFT_FEM simulation pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the PFT_FEM CLI.

    Args:
        args: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        prog="pft-simulate",
        description="Simulate MRI images with tumor growth in the posterior fossa",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./output"),
        help="Output directory for simulation results",
    )

    parser.add_argument(
        "-a", "--atlas",
        type=Path,
        default=None,
        help="Path to SUIT atlas directory (uses synthetic if not provided)",
    )

    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=365.0,
        help="Simulation duration in days (default: 365)",
    )

    parser.add_argument(
        "--tumor-center",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Tumor seed center in mm",
    )

    parser.add_argument(
        "--tumor-radius",
        type=float,
        default=2.5,
        help="Initial tumor radius in mm (default: 2.5)",
    )

    parser.add_argument(
        "--proliferation-rate",
        type=float,
        default=0.012,
        help="Tumor proliferation rate (1/day, default: 0.012)",
    )

    parser.add_argument(
        "--diffusion-rate",
        type=float,
        default=0.15,
        help="Tumor diffusion rate (mm^2/day, default: 0.15)",
    )

    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=["T1", "T2", "FLAIR", "T1_contrast", "DWI"],
        default=["T1", "T2", "FLAIR"],
        help="MRI sequences to generate",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    parsed_args = parser.parse_args(args)

    # Run simulation
    try:
        return run_simulation(parsed_args)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_simulation(args: argparse.Namespace) -> int:
    """Run the simulation with parsed arguments."""
    import time
    from .atlas import SUITAtlasLoader
    from .simulation import MRISimulator, TumorParameters, MRISequence
    from .io import NIfTIWriter

    start_time = time.time()

    if args.verbose:
        print("=" * 70)
        print("  PFT_FEM: Posterior Fossa Tumor Finite Element Modeling")
        print("=" * 70)
        print()

    # =========================================================================
    # STEP 1: Load Atlas
    # =========================================================================
    if args.verbose:
        print("[Step 1/5] Loading Atlas")
        print("-" * 70)
        print(f"  Source: {args.atlas or 'synthetic (built-in)'}")

    step_start = time.time()
    loader = SUITAtlasLoader(args.atlas)
    atlas_data = loader.load()

    if args.verbose:
        elapsed = time.time() - step_start
        print(f"  Shape: {atlas_data.shape}")
        print(f"  Voxel size: {atlas_data.voxel_size} mm")
        print(f"  Regions: {len(atlas_data.regions)} anatomical labels")
        print(f"  Completed in {elapsed:.2f}s")
        print()

    # =========================================================================
    # STEP 2: Configure Parameters
    # =========================================================================
    tumor_params = TumorParameters(
        center=tuple(args.tumor_center),
        initial_radius=args.tumor_radius,
        proliferation_rate=args.proliferation_rate,
        diffusion_rate=args.diffusion_rate,
    )

    if args.verbose:
        print("[Step 2/5] Configuring Simulation Parameters")
        print("-" * 70)
        print("  Tumor Parameters:")
        print(f"    Center location:     ({tumor_params.center[0]:.1f}, {tumor_params.center[1]:.1f}, {tumor_params.center[2]:.1f}) mm")
        print(f"    Initial radius:      {tumor_params.initial_radius:.1f} mm")
        print(f"    Initial density:     {tumor_params.initial_density:.1f}")
        print(f"    Proliferation rate:  {tumor_params.proliferation_rate:.4f} /day")
        print(f"    Diffusion rate:      {tumor_params.diffusion_rate:.3f} mm²/day")
        print(f"    Necrotic threshold:  {tumor_params.necrotic_threshold:.1f}")
        print(f"    Edema extent:        {tumor_params.edema_extent:.1f} mm")
        print()
        print("  Biophysical Constraints:")
        print("    Young's modulus:     3000 Pa (brain tissue)")
        print("    Poisson ratio:       0.45 (nearly incompressible)")
        print("    Boundary condition:  Fixed at domain edges")
        print()

    # Create simulator
    simulator = MRISimulator(atlas_data, tumor_params)

    # Parse sequences
    sequences = [MRISequence[seq] for seq in args.sequences]

    if args.verbose:
        print("  MRI Sequences to Generate:")
        for seq in sequences:
            print(f"    - {seq.value}")
        print()

    # =========================================================================
    # STEP 3: Generate Mesh & Setup FEM
    # =========================================================================
    if args.verbose:
        print("[Step 3/5] Generating Finite Element Mesh")
        print("-" * 70)

    step_start = time.time()
    simulator.setup()

    if args.verbose:
        elapsed = time.time() - step_start
        print(f"  Mesh Statistics:")
        print(f"    Nodes:               {simulator.mesh.num_nodes:,}")
        print(f"    Tetrahedral elements:{simulator.mesh.num_elements:,}")
        print(f"    Boundary nodes:      {len(simulator.mesh.boundary_nodes):,}")
        print(f"  FEM Matrices Built:")
        print(f"    Mass matrix:         {simulator.mesh.num_nodes}×{simulator.mesh.num_nodes}")
        print(f"    Stiffness matrix:    {simulator.mesh.num_nodes*3}×{simulator.mesh.num_nodes*3}")
        print(f"    Diffusion matrix:    {simulator.mesh.num_nodes}×{simulator.mesh.num_nodes}")
        print(f"  Completed in {elapsed:.2f}s")
        print()

    # =========================================================================
    # STEP 4: Run Tumor Growth Simulation
    # =========================================================================
    if args.verbose:
        print(f"[Step 4/5] Simulating Tumor Growth ({args.duration:.0f} days)")
        print("-" * 70)
        print("  Progress:")

    step_start = time.time()
    num_steps = int(args.duration)

    def progress_callback(state, step_idx):
        if args.verbose:
            vol = simulator.solver.compute_tumor_volume(state)
            disp = simulator.solver.compute_max_displacement(state)
            progress = (step_idx / num_steps) * 100
            bar_width = 30
            filled = int(bar_width * step_idx / num_steps)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r    Day {state.time:5.1f} [{bar}] {progress:5.1f}%  "
                  f"Volume: {vol:8.1f} mm³  Displacement: {disp:5.2f} mm", end="")

    # Run with custom callback for detailed progress
    states = simulator.solver.simulate(
        initial_state=simulator._create_initial_state(),
        duration=args.duration,
        dt=1.0,
        callback=progress_callback if args.verbose else None,
    )

    if args.verbose:
        print()  # New line after progress bar
        elapsed = time.time() - step_start
        final_state = states[-1]
        final_volume = simulator.solver.compute_tumor_volume(final_state)
        final_disp = simulator.solver.compute_max_displacement(final_state)
        print(f"  Final Results:")
        print(f"    Tumor volume:        {final_volume:.2f} mm³")
        print(f"    Max displacement:    {final_disp:.2f} mm")
        print(f"    Time steps computed: {len(states)}")
        print(f"  Completed in {elapsed:.2f}s")
        print()

    # =========================================================================
    # STEP 5: Generate MRI Images & Save Results
    # =========================================================================
    if args.verbose:
        print("[Step 5/5] Generating MRI Images & Saving Results")
        print("-" * 70)

    step_start = time.time()
    final_state = states[-1]

    if args.verbose:
        print("  Generating MRI sequences...")
    mri_images = simulator.generate_mri(final_state, sequences)

    if args.verbose:
        for seq_name in mri_images:
            print(f"    Generated: {seq_name}")

    # Create masks and deformed atlas
    if args.verbose:
        print("  Creating segmentation masks...")
    tumor_density = simulator._interpolate_to_volume(final_state.cell_density)
    tumor_mask = tumor_density > 0.1

    from scipy import ndimage
    dilated = ndimage.binary_dilation(tumor_mask, iterations=5)
    edema_mask = dilated & ~tumor_mask & (atlas_data.labels > 0)

    if args.verbose:
        print("  Applying spatial deformation to atlas...")
    deformed_atlas = simulator._apply_deformation(
        atlas_data.template.copy(),
        final_state
    )

    # Assemble result
    from .simulation import SimulationResult
    tumor_volume = simulator.solver.compute_tumor_volume(final_state)
    max_displacement = simulator.solver.compute_max_displacement(final_state)

    result = SimulationResult(
        tumor_states=states,
        mri_images=mri_images,
        deformed_atlas=deformed_atlas,
        tumor_mask=tumor_mask,
        edema_mask=edema_mask,
        metadata={
            "duration_days": args.duration,
            "final_tumor_volume_mm3": tumor_volume,
            "max_displacement_mm": max_displacement,
            "tumor_params": {
                "center": tumor_params.center,
                "initial_radius": tumor_params.initial_radius,
                "proliferation_rate": tumor_params.proliferation_rate,
                "diffusion_rate": tumor_params.diffusion_rate,
            },
            "atlas_shape": atlas_data.shape,
            "voxel_size": atlas_data.voxel_size,
        },
    )

    # Write results
    if args.verbose:
        print(f"  Writing output files to: {args.output}")

    writer = NIfTIWriter(
        output_dir=args.output,
        affine=atlas_data.affine,
        base_name="pft_simulation",
    )

    paths = writer.write_simulation_results(result)

    if args.verbose:
        elapsed = time.time() - step_start
        print("  Output Files:")
        for name, path in paths.items():
            print(f"    {name}: {path.name if hasattr(path, 'name') else path}")
        print(f"  Completed in {elapsed:.2f}s")
        print()

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    if args.verbose:
        print("=" * 70)
        print("  SIMULATION COMPLETE")
        print("=" * 70)
        print(f"  Total time:            {total_time:.2f}s")
        print(f"  Final tumor volume:    {result.metadata['final_tumor_volume_mm3']:.2f} mm³")
        print(f"  Max tissue displacement: {result.metadata['max_displacement_mm']:.2f} mm")
        print(f"  Output directory:      {args.output}")
        print("=" * 70)
    else:
        print(f"Simulation complete ({total_time:.1f}s). Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
