# PSMAReg — Quantification-Preserving Registration for Longitudinal Whole-Body PSMA PET/CT

Code for our submission to the **Learn2Reg 2026 PSMAReg** challenge: registration of
pre-therapy (baseline, *fixed*) and follow-up (*moving*) whole-body PSMA PET/CT scans
with an explicit constraint on PET-derived biomarkers.

<p align="center">
  <img src="assets/overview.png" alt="Overview of the registration pipeline" width="100%">
</p>

## Method

A three-stage pipeline built on the diffeomorphic, coarse-to-fine **LapIRN** architecture
([Mok & Chung, MICCAI 2020](https://doi.org/10.1007/978-3-030-59716-0_21); BibTeX under
[Citation](#citation)):

1. **Affine pre-registration.** ANTs estimates a CT-only affine transform per pair
   (body masks, CT windowed to [−1000, 1000] HU, half resolution, Mattes mutual
   information). It is converted to a dense field and applied to the moving scan.
2. **Deformable registration.** A three-level Laplacian pyramid (quarter, half and full
   resolution). Each level takes four channels — fixed CT, fixed PET, affinely aligned
   moving CT, moving PET — predicts a stationary velocity field, adds it to the
   upsampled velocity of the coarser level, and exponentiates it by scaling and squaring,
   which keeps the transform diffeomorphic. Levels are trained sequentially, each finer
   level initialised from the previous one. Image similarity is computed on **CT only**,
   since PET uptake legitimately changes after therapy; PET enters as network input and
   through the preservation losses below.
3. **Instance optimization (IO).** At test time each pair is refined by optimising a
   *residual* velocity field on top of the network prediction, initialised at zero and
   exponentiated the same way, so the refinement stays diffeomorphic. It uses the
   training objective; the best-scoring iterate (DSC, HD95, MTV, TLG) is kept.

The affine and the deformable field are composed into a single total transform.

## Losses

The same weighted objective drives training and instance optimization, in three groups —
one per evaluation criterion:

**Registration accuracy**
- `L_NCC` — local normalised cross-correlation on the CT channel, summed over as many
  scales as the pyramid level has resolutions (one at the coarsest, three at full res).
- `L_DSC` — Dice on the CT organ labels (weak supervision).

**PET quantification** (all evaluated on the composed transform and the original moving
lesion mask)
- `L_MTV` — relative change in metabolic tumor volume.
- `L_MTV-J` — deviation of the *mean* Jacobian determinant over the warped lesion from 1.
- `L_jac` — per-voxel deviation of the Jacobian determinant from 1, which discourages
  locally compensating compression and expansion.
- `L_TLG` — relative change in PET intensity mass inside the lesion (total lesion
  glycolysis), capturing both volume change and intensity interpolation.

**Deformation regularization**
- `L_rigid` — per-structure rigidity. Each of the 61 skeletal TotalSegmentator labels is
  fitted independently with a closed-form Kabsch rigid transform and the residual is
  penalised, constraining deformation *within* a bone while structures stay free to move
  relative to each other.
- `L_smooth` — diffusion regularizer (mean squared spatial gradient of the displacement).
- `L_NDV` — a differentiable reproduction of the challenge's non-diffeomorphic-volume
  metric (six-tetrahedra decomposition, negative Jacobian parts accumulated over the body
  mask), so gradients appear exactly where folding occurs.

Only the full-resolution level is trained with the complete objective. The two coarser
levels use similarity, label and regularization terms only — downsampled lesions and thin
ribs span too few voxels for volume ratios and per-structure rigid fits to be meaningful.

## Inputs and outputs

**Input** — four NIfTI volumes per pair, as released by the challenge:

| | file | role |
|---|---|---|
| fixed CT | `PSMARegPSMA_XXXX_0000_00.nii.gz` | baseline |
| fixed PET | `PSMARegPSMA_XXXX_0001_00.nii.gz` | baseline |
| moving CT | `PSMARegPSMA_XXXX_0000_01.nii.gz` | follow-up |
| moving PET | `PSMARegPSMA_XXXX_0001_01.nii.gz` | follow-up |

All volumes are 192 × 192 × 288 voxels at 2.7344 × 2.7344 × 3.27 mm; no further geometric
preprocessing is applied. Intensities are clipped (CT to [−1000, 1500] HU, PET SUV to
[0, 20]) and rescaled to [0, 1] internally.

**Output** — one dense displacement field per pair, `disp_XXXX_00_XXXX_01.nii.gz`:
channel-first `(3, 192, 192, 288)`, in **voxel units**, mapping the moving (follow-up)
scan into the fixed (baseline) frame.

Training additionally consumes the CT organ labels and PET lesion labels. For cases
without released labels these are generated: TotalSegmentator in fast mode for CT
organs, and an nnU-Net trained on this cohort for PET lesions.

## Results

Official Learn2Reg 2026 evaluation-server numbers. The hidden **test** results are not
available yet and will be added here once the organizers release them.

| leaderboard | rank | DSC (%) ↑ | HD95 (mm) ↓ | MTV error (%) ↓ | TLG error (%) ↓ | NDV (%) ↓ |
|---|---|---|---|---|---|---|
| validation | 7/45 | 78.9 ± 3.8 | 5.99 ± 2.39 | 1.56 ± 1.22 | 4.18 ± 2.49 | 0.0041 ± 0.0024 |
| **test** | — | — | — | — | — | — |

## What is in this repository

| path | what it is |
|---|---|
| `train.py` | training entry point — runs the pyramid levels sequentially |
| `inference.py` | inference on a single pair: affine → network → optional IO → displacement field |
| `psmareg/` | the method itself: model, losses, data pipeline, affine stage, instance optimization, config |
| `requirements.txt` | Python dependencies |
| `assets/` | figures |

The **submission container image is not in the repository** — it bundles the segmentation
weights and is several GB, so it is published to the GitHub Container Registry as
[`ghcr.io/koegl/psmareg`](https://github.com/users/koegl/packages/container/package/psmareg).

## Installation

Python 3.11, CUDA 12.x, one GPU with ≥ 24 GB VRAM for inference.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.6.0` (cu124). The pin matters: `antspyx` otherwise
pulls a cu13 wheel, which cannot initialise CUDA on a 12.x driver. TotalSegmentator and
nnU-Net are in there too; they are only touched when instance optimization has to
generate its own labels.

Tested with torch 2.6.0+cu124, numpy 2.3.5, scipy 1.15.3, nibabel 5.4.2, antspyx 0.6.3
and nnunetv2 2.8.1 on an NVIDIA RTX A6000.

## Running the container

Pull the image (no login needed, the package is public):

```bash
docker pull ghcr.io/koegl/psmareg:v1.0.0
```

Then register one pair — five positional paths, in this order (fixed CT, fixed PET,
moving CT, moving PET, output field):

```bash
docker run --rm --ipc=host --memory 60g --gpus "device=0" --user $(id -u):$(id -g) --network=none --mount type=bind,source=<image dir>,target=/app/input,readonly --mount type=bind,source=<output dir>,target=/app/output ghcr.io/koegl/psmareg:v1.0.0 /app/input/PSMARegPSMA_0001_0000_00.nii.gz /app/input/PSMARegPSMA_0001_0001_00.nii.gz /app/input/PSMARegPSMA_0001_0000_01.nii.gz /app/input/PSMARegPSMA_0001_0001_01.nii.gz /app/output/disp_0001_00_0001_01.nii.gz
```

No other arguments are needed; every default is baked into the image. Resource envelope:
1 CUDA GPU (24 GB VRAM peak), ~8 GB RAM, scales to ~29 CPU cores, **~90 s per pair** — the
container holds an internal 90 s wall-clock budget and instance optimization takes as many
steps as fit inside it, so a faster machine takes more steps rather than finishing sooner.

The image is self-contained: model weights, the PET lesion nnU-Net and the
TotalSegmentator weights are all baked in, and it runs with `--network=none`.

## Training

Data is expected in the layout released by the challenge. Levels are trained in order,
each initialised from the previous one:

```bash
python train.py --data-dir /path/to/PSMAReg_dataset --out-dir runs/psmareg --level 1
```

```bash
python train.py --data-dir /path/to/PSMAReg_dataset --out-dir runs/psmareg --level 2 --init runs/psmareg/lvl1_best.pth
```

```bash
python train.py --data-dir /path/to/PSMAReg_dataset --out-dir runs/psmareg --level 3 --init runs/psmareg/lvl2_best.pth
```

Loss weights, augmentation and the IO settings live in `psmareg/config.py`; all of them
can be overridden on the command line. Reference run: 60k / 60k / 120k steps with Adam at
3·10⁻⁴, 2·10⁻⁴ and 2.5·10⁻⁴, batch size 1 with gradient accumulation over 4 steps, five
epochs of linear warmup per level, the preceding level frozen for the first ten epochs.
The convolutional trunk runs in bfloat16 while transforms and losses stay in fp32, to keep
scaling-and-squaring and the Jacobian terms numerically sound. All three levels take
≈ 2 d 6 h on a single NVIDIA H100 80 GB.

## Inference

Without the container, on a single pair:

```bash
python inference.py --fixed-ct fixed_ct.nii.gz --fixed-pet fixed_pet.nii.gz --moving-ct moving_ct.nii.gz --moving-pet moving_pet.nii.gz --weights model.pth --out disp.nii.gz
```

Affine pre-registration, the network, and the composition of the two: about 21 s per
pair on an RTX A6000, most of it the CPU-bound ANTs affine.

### Instance optimization

`--io` refines the field for that one pair. It needs a PET lesion mask of the moving
scan and CT organ labels for both scans. Supply them directly:

```bash
python inference.py ... --io --moving-lesion lesion.nii.gz --moving-labels moving_labels.nii.gz --fixed-labels fixed_labels.nii.gz
```

or let them be predicted — CT organs from TotalSegmentator, PET lesions from an nnU-Net
model directory:

```bash
python inference.py ... --io --lesion-model /path/to/Dataset501_PSMALesion/nnUNetTrainer_PGPSplus__nnUNetPlans__3d_fullres
```

Each label gates exactly one group of terms, so anything missing simply switches those
off rather than failing: without the lesion mask the PET terms go, without the CT
labels the Dice and rigidity terms go, and with neither the refinement runs on NCC,
smoothness and the folding barrier alone. `--io-steps` and `--io-lr` tune the loop;
the remaining weights live in `IOConfig` in [psmareg/instance_opt.py](psmareg/instance_opt.py).

### PET lesion model

`--lesion-model` is an nnU-Net results folder — the one holding `plans.json`,
`dataset.json` and `fold_0/` — containing the model the container runs: an nnU-Net
trained on the challenge cohort with progressive growing of patch size
([Fischer et al.](https://arxiv.org/abs/2407.07853)), `nnUNetTrainer_PGPSplus`.

That trainer changes only the training schedule, never the architecture, so stock
nnU-Net loads the checkpoint once the trainer name resolves — which is all
`_patch_trainer_lookup` in [psmareg/segmentation.py](psmareg/segmentation.py) arranges.
The training fork is not needed at inference.

Use this model rather than another lesion segmenter if you are reproducing the
submission: the MTV and TLG terms are computed on its mask, so a different mask gives a
different refinement. One pair takes about 17 s. `--lesion-folds 0 1 2 3 4` ensembles
five folds where a model has them, at five times the cost.

The affine stage is not seeded from Python (ITK uses its own RNG), so repeated runs on
the same pair differ slightly — on the order of 0.1 voxels on average.

## Docker prerequisites

Docker Engine and the NVIDIA Container Toolkit (for `--gpus`) must be installed:

- Docker Engine: <https://docs.docker.com/engine/install/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

Check both with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`,
which should print your GPU.

## Citation

```bibtex
@inproceedings{psmareg2026,
  title     = {PET Quantification-Preserving Registration for Longitudinal Whole-Body PSMA PET/CT},
  author    = {Anonymous},
  booktitle = {Learn2Reg 2026},
  year      = {2026}
}
```

This work builds on LapIRN, which should be cited alongside it:

```bibtex
@InProceedings{10.1007/978-3-030-59716-0_21,
  author    = {Mok, Tony C. W. and Chung, Albert C. S.},
  editor    = {Martel, Anne L. and Abolmaesumi, Purang and Stoyanov, Danail
               and Mateus, Diana and Zuluaga, Maria A. and Zhou, S. Kevin
               and Racoceanu, Daniel and Joskowicz, Leo},
  title     = {Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020},
  year      = {2020},
  publisher = {Springer International Publishing},
  address   = {Cham},
  pages     = {211--221},
  isbn      = {978-3-030-59716-0}
}
```
