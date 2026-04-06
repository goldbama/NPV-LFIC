# NPV-LFIC: Neural Pseudo-Video Light Field Image Compression

**"Neural Pseudo-Video Light Field Image Compression via Adaptive Sparse Mode Selection and Frame Interpolation Reconstruction"**

The method decomposes light field image compression into three coordinated modules:

1. `OSMS-Net` for adaptive sparse mode selection
2. `DCVC-LFI` for pseudo-video key-view (and residual) compression
3. `BIM-VFI` for progressive interpolation-based view reconstruction

## Highlights

- Knowledge-guided sparse mode decision instead of brute-force mode search
- LFI-oriented pseudo-video coding with neural video compression
- Progressive interpolation reconstruction with optional ACE residual enhancement

## Repository Structure

```text
NPV-LFIC_code/
|-- OSMS-Net/
|   |-- test.py
|   |-- csv/
|   |-- png/
|   `-- checkpoints/
|-- DCVC-LFI/
|   |-- test_video_single_bitrate.py
|   |-- test_video_single_bitrate_res.py
|   |-- dataset_config_example_9x9_all.json
|   |-- dataset_config_example_9x9_q4_res_Mask_0.5dB_SnakeRev_ALL.json
|   |-- src/
|   |-- media/
|   `-- checkpoints/
|-- BIM-VFI/
|   |-- main.py
|   |-- infer_all_modes_EPFL.py
|   |-- cfgs/
|   |-- modules/
|   `-- pretrained/
```

## Method Overview

For a 9x9 light field image:

1. `OSMS-Net` predicts one of six preset sparse modes (`5/9/13/25/41/81` views) from five representative views + target bitrate level (`Q1-Q4`).
2. Key views are arranged as a snake-order pseudo-video and compressed by `DCVC-LFI`.
3. Missing views are progressively reconstructed with `BIM-VFI` via midpoint interpolation.
4. For medium/high bitrate points (`Q2-Q4`), ACE residual coding can be enabled to improve angular quality consistency.

## Environment

- NVIDIA GPU (CUDA-capable)
- Paper training/testing platform: Intel i7-12700 + RTX 4070 Ti Super

Notes:

- `BIM-VFI` uses custom CUDA/CuPy kernels (`costvol.py`), so CuPy is required for interpolation inference.
- Some dependencies are only needed for training/benchmark branches.

## Checkpoints

Please place pretrained weights before running:

- `OSMS-Net/checkpoints/checkpoint_best_31.pth` 
- `DCVC-LFI/checkpoints/acmmm2022_image_psnr.pth.tar`
- `DCVC-LFI/checkpoints/q1.tar`, `q2.tar`, `q3.tar`, `q4.tar` 
- `BIM-VFI/pretrained/bim_vfi.pth`

**Pretrained Weights Download**

- Baidu download link: https://pan.baidu.com/s/1MT1wtjyA2XOQAkd3Nuj0Jg?pwd=zjgh

- Google download link: https://drive.google.com/drive/folders/1JuuYJgJLBEvLWCc1BHjeLIyJWeZhmZdH?usp=drive_link

## Data Layout and Naming

### A) OSMS-Net input layout

`OSMS-Net/test.py` expects scene folders with angular view naming such as:

- `000_000.png`, `000_004.png`, `004_004.png`, `008_004.png`, etc.

It reads 5 representative views:

- `004_004.png` (center)
- `000_004.png` (top)
- `008_004.png` (bottom)
- `004_000.png` (left)
- `004_008.png` (right)

### B) DCVC-LFI key-view pseudo-video layout

`test_video_single_bitrate.py` expects each sequence as ordered PNG frames:

- `im00001.png`, `im00002.png`, ...

Dataset metadata is described by:

- `dataset_config_example_9x9_all.json`

### C) Residual sequence layout (ACE branch)

`test_video_single_bitrate_res.py` supports coordinate-style naming:

- `000_008_006.png`, `001_008_005.png`, ...

and uses:

- `dataset_config_example_9x9_q4_res_Mask_0.5dB_SnakeRev_ALL.json`

## How To Run

### 1) Evaluate OSMS-Net mode prediction

```bash
cd OSMS-Net
python test.py
```

Output:

- Log file: `OSMS-Net/checkpoints/test_run.log`
- Predicted mode per scene/rate and overall accuracy

### 2) Run base-layer pseudo-video coding (DCVC-LFI)

```bash
cd DCVC-LFI
python test_video_single_bitrate.py \
  --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar \
  --model_path ./checkpoints/q4.tar \
  --rate_num 1 \
  --rate_idx 3 \
  --test_config ./dataset_config_example_9x9_all.json \
  --cuda 1 \
  -w 1 \
  --write_stream 1 \
  --output_path ./results/DCVC-LFI_9x9_ALL.json \
  --save_decoded_frame true
```

Main outputs:

- Metrics JSON: `--output_path`
- Bitstream bins: `out_bin/<scene>/<rate_idx>/*.bin`
- Decoded frames (if enabled): `experiments/<model_name>/save_png/<dataset>/<scene>/`

### 3) Run interpolation reconstruction (BIM-VFI)

Batch interpolation script for predefined sparse modes (`EPFL_5/9/13/25/41`):

```bash
cd BIM-VFI
python infer_all_modes_EPFL.py
```

Before batch run, edit `BASE_ROOT_DIR` in `infer_all_modes_EPFL.py` to your decoded key-view root.

### 4) Optional ACE residual coding

```bash
cd DCVC-LFI
python test_video_single_bitrate_res.py \
  --i_frame_model_path ./checkpoints/acmmm2022_image_psnr.pth.tar \
  --model_path ./checkpoints/q3.tar \
  --rate_num 1 \
  --rate_idx 2 \
  --test_config ./dataset_config_example_9x9_q4_res_Mask_0.5dB_SnakeRev_ALL.json \
  --cuda 1 \
  -w 1 \
  --write_stream 1 \
  --output_path ./results/DCVC-LFI_residual.json \
  --save_decoded_frame true \
  --force_intra_period 9999
```

## Acknowledgement

This project builds on or adapts components from prior open-source works, including:

- DCVC/DCVC-HEM related implementations
- BiM-VFI interpolation network
