# Pullback-Guided Diffusion

Research code for diversity-oriented conditional pullback sampling. The
repository compares four sampling methods with identical initial latents:

- Clean DDIM
- CADS
- TPSO
- Rho-Star Pullback (ours)

Both text-to-image generation and BrushNet inpainting support Stable Diffusion
1.5 and Stable Diffusion XL.

## Quick start

### Requirements

- Linux and Git
- Conda or Miniforge
- Python 3.10
- an NVIDIA GPU with CUDA support
- approximately 15 GB of disk space for one SDXL workflow

The SDXL experiments were developed on an NVIDIA A100 40 GB. Smaller GPUs may
require fewer particles or smaller U-Net microbatches.

### 1. Clone the repository

```bash
git clone https://github.com/syldeha/PullBack-directions-for-diversity.git
cd PullBack-directions-for-diversity
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate pullback
export PYTHONNOUSERSITE=1
```

`environment.yml` installs the pinned dependencies from
`requirements-lock.txt`. A less restrictive `requirements.txt` is also
provided for development.

### 3. Install the BrushNet Diffusers fork

The repository uses the custom Diffusers implementation released with
BrushNet. Install the tested revision:

```bash
export BRUSHNET_ROOT="$HOME/BrushNet"
git clone https://github.com/TencentARC/BrushNet.git "$BRUSHNET_ROOT"
git -C "$BRUSHNET_ROOT" checkout 0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d
python -m pip install --no-build-isolation --no-deps -e "$BRUSHNET_ROOT"
```

Do not install the public `diffusers` package afterward because it replaces
the BrushNet pipeline classes. This command should print a path inside
`$BRUSHNET_ROOT/src/diffusers`:

```bash
python -c "import diffusers; print(diffusers.__file__)"
```

### 4. Download model assets

Accept the relevant Hugging Face model licenses, then authenticate if needed:

```bash
huggingface-cli login
```

Choose one workflow profile:

| Profile | Task | Backbone | Additional data |
|---|---|---|---|
| `generation-sd15` | Text-to-image | Stable Diffusion 1.5 | None |
| `generation-sdxl` | Text-to-image | Stable Diffusion XL | None |
| `inpainting-sd15` | BrushNet inpainting | Stable Diffusion 1.5 | BrushBench |
| `inpainting-sdxl` | BrushNet inpainting | Stable Diffusion XL | BrushBench |

For example, download SDXL inpainting weights and the CLIP/DINO models used by
Rho-Star selection:

```bash
python -m scripts.download_assets \
  --profile inpainting-sdxl \
  --include-metrics
```

Replace the profile name to download another workflow. Assets are stored in:

```text
$BRUSHNET_ROOT/checkpoints/
|-- stable-diffusion-v1-5/
|-- stable-diffusion-xl-base-1.0/
|-- brushnet_segmentation_mask/
`-- brushnet_sdxl/
```

Custom checkpoint locations can be supplied with `--base-model` and
`--brushnet-model`.

### 5. Add BrushBench for inpainting

BrushBench is license-controlled and is not downloaded automatically. Follow
the [upstream BrushNet data instructions](https://github.com/TencentARC/BrushNet#data-download-%EF%B8%8F),
then point this repository to the extracted directory:

```bash
export BRUSHBENCH_ROOT=/absolute/path/to/BrushBench
```

Expected layout:

```text
BrushBench/
|-- mapping_file.json
`-- images/
    |-- 000000000.jpg
    `-- ...
```

The default location is `$BRUSHNET_ROOT/data/BrushBench`.

## Run a quick comparison

The comparison command runs the selected methods sequentially using the same
initial latents. It defaults to four images per method and ten DDIM steps.

For text-to-image generation:

```bash
python -m scripts.run_comparison \
  --profile generation-sdxl \
  --prompt "A red fox standing in an autumn forest, photorealistic"
```

For BrushNet inpainting:

```bash
python -m scripts.run_comparison \
  --profile inpainting-sdxl \
  --example 000000089 \
  --brushbench-root "$BRUSHBENCH_ROOT"
```

Useful options:

```text
--methods all                  # or clean_ddim,cads,tpso,rho_star
--num-particles 4
--steps 10
--output-dir /custom/output
```

Results are written to `outputs/comparisons/<profile>/`:

```text
comparison.png
comparison.pdf
result.json
clean_ddim/p00.png ...
cads/p00.png ...
tpso/p00.png ...
rho_star/p00.png ...
```

The quick comparison uses reduced pullback ranks and short TPSO optimization
so a user can inspect the complete workflow quickly. It is not the paper
benchmark; use the configurations in `configs/` for scientific results.

## Interactive notebooks

Two notebooks expose the same comparison workflows interactively:

- `notebooks/generation_comparison.ipynb`
- `notebooks/inpainting_comparison.ipynb`

Start Jupyter from the repository root:

```bash
jupyter lab
```

For inpainting, export `BRUSHBENCH_ROOT` and any custom model paths before
starting Jupyter. Restart the kernel after changing the model family or
checkpoint paths.

## Full experiments

The long runners add datasets, metrics, resumable outputs, aggregate tables,
and optional W&B logging.

### Text-to-image generation

Prepare MS COCO 2017:

```bash
export COCO_ROOT=/absolute/path/to/coco2017
python -m scripts.prepare_coco2017
```

Download the SSCD metric checkpoint:

```bash
mkdir -p "$HOME/models"
wget -O "$HOME/models/sscd_disc_mixup.torchscript.pt" \
  https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
export SSCD_MODEL_PATH="$HOME/models/sscd_disc_mixup.torchscript.pt"
```

Edit `configs/generation.py` to select the model family, prompts, particle
count, pullback ranks, rho candidates, output name, and W&B settings. Then run:

```bash
python -m experiments.generation --check
python -u -m experiments.generation
```

SD1.5 and SDXL require different perturbation magnitudes. Typical SD1.5 rho
values are approximately `0.10` to `0.25`; the current SDXL configuration uses
larger values. Retune when changing model families.

### BrushNet inpainting

Set the model family and paths before launching the runner. For SDXL:

```bash
export BRUSHNET_ROOT="$HOME/BrushNet"
export BRUSHNET_MODEL_FAMILY=sdxl
export BRUSHNET_BASE_MODEL="$BRUSHNET_ROOT/checkpoints/stable-diffusion-xl-base-1.0"
export BRUSHNET_MODEL="$BRUSHNET_ROOT/checkpoints/brushnet_sdxl"
export BRUSHBENCH_ROOT=/absolute/path/to/BrushBench
```

For SD1.5:

```bash
export BRUSHNET_MODEL_FAMILY=sd15
export BRUSHNET_BASE_MODEL="$BRUSHNET_ROOT/checkpoints/stable-diffusion-v1-5"
export BRUSHNET_MODEL="$BRUSHNET_ROOT/checkpoints/brushnet_segmentation_mask"
```

Edit `configs/inpainting.py` to select examples, methods, perturbation
magnitudes, output name, and W&B settings. Then run:

```bash
python -m experiments.inpainting --check
python -u -m experiments.inpainting
```

Rebuild tables from completed records without loading the model:

```bash
python -m experiments.inpainting --aggregate-only
```

### Resume and W&B

Every completed `(example, method)` result is saved immediately. Re-run the
same command with the same configuration and `RUN_NAME` to resume. Change
`RUN_NAME` whenever a model or scientific parameter changes; incompatible
saved configurations are rejected.

Authenticate for online W&B logging:

```bash
wandb login
```

Use local/offline logging with:

```bash
export WANDB_MODE=offline
```

Local result files do not depend on W&B.

## Numerical precision

The SDXL VAE can overflow in half precision. The production encode/decode
paths temporarily run force-upcast VAEs in `float32`, restore their original
dtype afterward, and reject NaN or Inf tensors before images are saved.

If CUDA runs out of memory, reduce the particle count. For full generation,
also reduce `UNET_PARTICLE_BATCH_SIZE` in `configs/generation.py`.

## Method summary

- **Clean DDIM** uses independent Gaussian particles with unchanged CFG
  conditions.
- **CADS** injects and rescales condition noise during the configured reverse
  diffusion window.
- **TPSO** optimizes particle-specific positive content-token offsets while
  keeping the text encoder and diffusion model frozen.
- **Rho-Star Pullback** estimates dominant condition-response directions with
  matrix-free products, probes candidate perturbation magnitudes under a CLIP
  constraint, and selects them for DINO diversity.

Implementation and metric details are available in
`docs/METHODOLOGY.md` and `docs/REPRODUCIBILITY.md`.

## Repository layout

```text
PullBack-directions-for-diversity/
|-- configs/       # Full experiment configurations
|-- generation/    # SD1.5/SDXL text-to-image pipeline
|-- inpainting/    # BrushNet preparation and samplers
|-- pullback/      # Matrix-free bases, directions, and precision handling
|-- methods/       # CADS, TPSO, adaptive pullback, and Rho-Star
|-- evaluation/    # Metrics
|-- experiments/   # Resumable full runners
|-- notebooks/     # Two interactive comparison notebooks
|-- scripts/       # Asset, comparison, and dataset commands
`-- docs/          # Method and reproducibility details
```
