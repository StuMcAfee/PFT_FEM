# HCP1065 DTI Atlas

This directory contains diffusion tensor imaging (DTI) templates derived from the Human Connectome Project (HCP) 1065-subject dataset.

## Files

| File | Description |
|------|-------------|
| `FSL_HCP1065_FA_1mm.nii.gz` | Fractional Anisotropy (FA) map |
| `FSL_HCP1065_V1_1mm.nii.gz` | Principal eigenvector (V1) - fiber orientation |

## File Details

### Fractional Anisotropy (FA)
- Scalar map indicating degree of anisotropic diffusion
- Range: 0 (isotropic) to 1 (fully anisotropic)
- Higher values indicate organized white matter tracts

### Principal Eigenvector (V1)
- 4D volume (x, y, z, 3) containing the primary diffusion direction
- Each voxel contains a 3D unit vector pointing along the main fiber direction
- Used for modeling anisotropic diffusion in white matter

## Source

From the [swarrington1/WM_atlases](https://github.com/swarrington1/WM_atlases) repository.

Original data from FSL (FMRIB Software Library), derived from 1065 Human Connectome Project subjects.

Reference: Warrington S, Jbabdi S, Smith S, Sotiropoulos S. HCP1065 DTI templates.

## Resolution

- Voxel size: 1x1x1 mm
- Space: MNI152 (ICBM 2009a Nonlinear Symmetric)

## License

Distributed under the FSL license. See https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Licence

## Usage

These files can be used for:
- Modeling anisotropic diffusion along white matter fiber tracts
- Defining fiber orientation for biophysical tumor growth models
- Computing tissue mechanical properties based on fiber direction
