# MNI152 Tissue Segmentation Atlas

This directory contains MNI152 template files with tissue probability maps from FSL FAST segmentation.

## Files

| File | Description |
|------|-------------|
| `MNI152_T1_1mm_Brain.nii.gz` | T1-weighted brain template (skull-stripped) |
| `MNI152_T1_1mm_Brain_Mask.nii.gz` | Binary brain mask |
| `MNI152_T1_1mm_Brain_FAST_seg.nii.gz` | Tissue segmentation (1=CSF, 2=GM, 3=WM) |
| `MNI152_T1_1mm_Brain_FAST_pve_0.nii.gz` | CSF probability map |
| `MNI152_T1_1mm_Brain_FAST_pve_1.nii.gz` | Gray matter probability map |
| `MNI152_T1_1mm_Brain_FAST_pve_2.nii.gz` | White matter probability map |

## Segmentation Labels

- **0**: Background
- **1**: Cerebrospinal Fluid (CSF)
- **2**: Gray Matter (GM)
- **3**: White Matter (WM)

## Source

From the [Jfortin1/MNITemplate](https://github.com/Jfortin1/MNITemplate) repository.

Original data from FSL (FMRIB Software Library) FAST tissue segmentation of the MNI152 template.

## Resolution

- Voxel size: 1x1x1 mm
- Dimensions: 182x218x182

## Usage

These files can be used for:
- Tissue-specific material properties in FEM simulation
- Defining boundary conditions based on tissue type
- Constraining tumor growth based on tissue barriers (e.g., CSF)
