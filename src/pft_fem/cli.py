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
        default=30.0,
        help="Simulation duration in days (default: 30)",
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
        default=5.0,
        help="Initial tumor radius in mm (default: 5)",
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
    from .atlas import SUITAtlasLoader
    from .simulation import MRISimulator, TumorParameters, MRISequence
    from .io import NIfTIWriter

    if args.verbose:
        print("PFT_FEM: Posterior Fossa Tumor Simulation")
        print("=" * 50)

    # Load atlas
    if args.verbose:
        print(f"Loading atlas from: {args.atlas or 'synthetic'}")

    loader = SUITAtlasLoader(args.atlas)
    atlas_data = loader.load()

    if args.verbose:
        print(f"  Atlas shape: {atlas_data.shape}")
        print(f"  Voxel size: {atlas_data.voxel_size}")

    # Configure tumor parameters
    tumor_params = TumorParameters(
        center=tuple(args.tumor_center),
        initial_radius=args.tumor_radius,
        proliferation_rate=args.proliferation_rate,
        diffusion_rate=args.diffusion_rate,
    )

    if args.verbose:
        print(f"\nTumor parameters:")
        print(f"  Center: {tumor_params.center}")
        print(f"  Initial radius: {tumor_params.initial_radius} mm")
        print(f"  Proliferation rate: {tumor_params.proliferation_rate} /day")
        print(f"  Diffusion rate: {tumor_params.diffusion_rate} mm²/day")

    # Create simulator
    simulator = MRISimulator(atlas_data, tumor_params)

    # Parse sequences
    sequences = [MRISequence[seq] for seq in args.sequences]

    if args.verbose:
        print(f"\nMRI sequences: {[s.value for s in sequences]}")
        print(f"Simulation duration: {args.duration} days")
        print("\nRunning simulation...")

    # Run simulation
    result = simulator.run_full_pipeline(
        duration_days=args.duration,
        sequences=sequences,
        verbose=args.verbose,
    )

    # Write results
    if args.verbose:
        print(f"\nWriting results to: {args.output}")

    writer = NIfTIWriter(
        output_dir=args.output,
        affine=atlas_data.affine,
        base_name="pft_simulation",
    )

    paths = writer.write_simulation_results(result)

    if args.verbose:
        print("\nOutput files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")

        print("\nSimulation summary:")
        print(f"  Final tumor volume: {result.metadata['final_tumor_volume_mm3']:.2f} mm³")
        print(f"  Max displacement: {result.metadata['max_displacement_mm']:.2f} mm")

    print(f"\nSimulation complete. Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
