# Pullback-Guided Diffusion

This repository contains the cleaned research implementation of conditional
pullback sampling for diffusion models. It supports two experiment tracks:

1. Full image generation with Stable Diffusion 1.5 or SDXL.
2. Image inpainting with BrushNet, with Stable Diffusion 1.5 or SDXL.

The code compares clean DDIM, CADS, TPSO, and pullback-guided sampling. The
full-generation track additionally implements particle-specific rho selection
with a short CLIP-constrained DINO probe.

This repository contains code only.

## Repository Layout

```text
pullback-diffusion/
|-- configs/       # Parameters edited before an experiment
|-- generation/    # SD1.5/SDXL loading, prompt encoding, CFG, and DDIM
|-- pullback/      # Matrix-free pullback basis and disjoint directions
|-- methods/       # CADS, TPSO, adaptive pullback, and rho-star (SD1.5/SDXL)
|-- inpainting/    # BrushNet preparation and all inpainting methods
|-- evaluation/    # Generation and inpainting metrics
|-- experiments/   # Resumable long-running experiment entry points
|-- notebooks/     # Small interactive comparisons
|-- scripts/       # Dataset preparation utilities
|-- tests/         # CPU checks for the numerical algorithms
`-- docs/          # Method and reproducibility details
```

The scientific code is separated from the evaluation machinery on purpose.
For example, the pullback basis is implemented in `pullback/basis.py`, while
`experiments/generation.py` only coordinates captions, checkpoints, metrics,
tables, and W&B logging.

## Implemented Methods

### Clean DDIM

The baseline uses independent initial Gaussian latents and the unmodified
positive and negative conditions during every CFG-DDIM step.

### CADS

The implementation follows the original condition-annealing experiment used
throughout this project:

- isotropic Gaussian noise is applied to both CFG branches;
- positive and negative condition noise is independent;
- each branch is rescaled against its own clean condition;
- the public configuration uses fresh condition noise at every active step;
- all methods still receive the exact same initial latent tensor.

### TPSO

TPSO optimizes one positive content-token offset per particle. BOS, EOS, and
padding remain fixed. The text encoder and diffusion model are frozen, the
negative CFG branch remains clean, and the optimized condition returns
linearly to the clean condition during sampling. For SDXL, offsets are
optimized jointly on both text encoders and their penultimate hidden states
are concatenated, matching `StableDiffusionXLPipeline.encode_prompt`.

### Adaptive Pullback

The method estimates dominant directions of the condition-response Gram
operator without constructing a full Jacobian:

```math
G_C v = J_C^T J_C v.
```

For regional inpainting direction discovery, it can instead use:

```math
G_{C,M}v = J_C^T W_M^T W_M J_Cv.
```

Centered finite differences compute `Jv`, autograd computes the VJP, and block
power iteration with Rayleigh-Ritz extraction recovers the requested basis.
Fixed Gaussian fields are projected into balanced disjoint basis subsets, so
the same particle identity persists when the basis is refreshed.

### Rho-Star

For full generation, rho-star selects one perturbation magnitude per particle.
At one intermediate timestep, it decodes only Tweedie candidate estimates,
rejects candidates whose CLIP score drops beyond the configured tolerance, and
jointly maximizes mean pairwise DINO cosine distance. The final trajectory is
then restarted from the original latent tensor with the selected scales.

The full equations and exact executable choices are in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Install

Python 3.10. Install the common dependencies with:

```bash
python -m pip install -r requirements.txt
```

BrushNet inpainting (both SD1.5 and SDXL) uses the custom Diffusers source
shipped by the upstream BrushNet repository. Clone and install it separately:

```bash
git clone https://github.com/TencentARC/BrushNet.git ~/BrushNet
cd ~/BrushNet
python -m pip install -e .
```

Everywhere below, `BRUSHNET_ROOT` defaults to `~/BrushNet` if left unset —
export it explicitly only if you cloned somewhere else:

```bash
export BRUSHNET_ROOT=/path/to/BrushNet
```

Follow the upstream BrushNet data and checkpoint licenses. They are not
redistributed here.

## Set Up Checkpoints

Every checkpoint below is placed at the exact local path the code already
defaults to, so once it's downloaded no further configuration is needed
unless noted otherwise.

### Stable Diffusion 1.5 (full generation)

No manual step needed. `stable-diffusion-v1-5/stable-diffusion-v1-5` is
downloaded automatically into the Hugging Face cache the first time
`experiments.generation` or `notebooks/generation_comparison.ipynb` runs.

### SDXL (full generation)

Full generation and BrushNet inpainting share the same SDXL backbone
checkpoint format. Download it once:

```bash
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ.get("BRUSHNET_ROOT", Path.home() / "BrushNet"))
snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir=str(root / "checkpoints" / "stable-diffusion-xl-base-1.0"),
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ],
)
PY
```

The patterns list exact fp16 filenames rather than a `*.safetensors`
wildcard on purpose: this repository hosts both fp32 and fp16 weights, and a
broad wildcard downloads both (~20 GB instead of ~6.5 GB) since only the
fp16 files are ever loaded.

This VAE overflows to NaN (solid black output) when decoded in plain fp16 —
it sets `force_upcast: true` in its own config to signal that. Both
`generation/model.py` and `inpainting/model.py`'s `decode_latents` check
that flag and temporarily decode in fp32, so this works correctly with no
extra configuration.

A community checkpoint such as
[RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)
works too — download it the same way with
`repo_id="RunDiffusion/Juggernaut-XL-v9"` and
`local_dir=".../checkpoints/JuggernautXL-v9"` (that folder name is
`inpainting/model.py`'s built-in default for `BRUSHNET_MODEL_FAMILY=sdxl`,
so using it needs no `BRUSHNET_BASE_MODEL` override; any other folder name,
including `stable-diffusion-xl-base-1.0` above, does).

### BrushNet adapter — SD1.5 (inpainting)

The SD1.5 BrushNet checkpoint (`segmentation_mask_brushnet_ckpt`) has no
Hugging Face mirror — like the SDXL adapter below, it is distributed only
inside a single shared Google Drive folder that also contains 10 unrelated
checkpoints
(https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n).
Download just the two required files by ID into the code's default path:

```bash
python -m pip install -U gdown

python - <<'PY'
import os
from pathlib import Path
import gdown

root = Path(os.environ.get("BRUSHNET_ROOT", Path.home() / "BrushNet"))
target = root / "checkpoints" / "brushnet_segmentation_mask"
target.mkdir(parents=True, exist_ok=True)

gdown.download(
    id="1z8jex6js-zzSxURnNOWMW1PP4njvtusE",
    output=str(target / "config.json"),
)
gdown.download(
    id="1uLQ55rdcFljdP_9iYGX5ZmJWzXFJast7",
    output=str(target / "diffusion_pytorch_model.safetensors"),
)
PY
```

These file IDs were read directly from the folder listing (see the SDXL
adapter section below for how to re-list it) but not independently
downloaded and verified in this pass — confirm them by re-running the
listing command if the download fails.

### BrushNet adapter — SDXL (inpainting)

This checkpoint (`segmentation_mask_brushnet_ckpt_sdxl_v1`) is in the same
Google Drive folder. Downloading the whole folder is impractical (11
unrelated checkpoints); list its contents without downloading anything to
get the file IDs instead:

```bash
python -m pip install -U gdown

python - <<'PY'
import gdown

files = gdown.download_folder(
    url="https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n",
    skip_download=True,
    quiet=True,
)
for f in files:
    if "segmentation_mask_brushnet_ckpt_sdxl_v1" in f.path:
        print(f.id, f.path)
PY
```

Then download just those two files by ID into the code's default adapter
path:

```bash
python - <<'PY'
import os
from pathlib import Path
import gdown

root = Path(os.environ.get("BRUSHNET_ROOT", Path.home() / "BrushNet"))
target = root / "checkpoints" / "brushnet_sdxl"
target.mkdir(parents=True, exist_ok=True)

# Use the IDs printed by the listing step above.
gdown.download(id="<config.json file id>", output=str(target / "config.json"))
gdown.download(
    id="<diffusion_pytorch_model.safetensors file id>",
    output=str(target / "diffusion_pytorch_model.safetensors"),
)
PY
```

As of this writing, `config.json` is file ID
`1jCt6KsP2jKSBD_QYJsuQ9Yb1rujLLt3f` and `diffusion_pytorch_model.safetensors`
is `1yPhfQ3y60fXURr9_YfQrdE48y68IZjBN` (downloaded and verified working in
this repository) — but Drive file IDs can change if the folder is
reorganized, so re-running the listing step above is the reliable way to
confirm them.

### Datasets and metric checkpoints

**BrushBench** (inpainting evaluation data):

```bash
export BRUSHBENCH_ROOT=/path/to/BrushBench
```

Follow the upstream BrushNet instructions to obtain the data; it is not
redistributed here.

**COCO 2017** (full-generation evaluation data). Choose the protocol in
`configs/coco2017.py`, then run:

```bash
export COCO_ROOT=/path/to/coco2017
python -m scripts.prepare_coco2017
```

The script downloads the official validation images and annotations when
they are missing, then builds the seeded caption manifest.

**SSCD** (full-generation metrics), Meta's official release for image copy
detection (`facebookresearch/sscd-copy-detection`):

```bash
mkdir -p /path/to/models
wget -O /path/to/models/sscd_disc_mixup.torchscript.pt \
  https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
export SSCD_MODEL_PATH=/path/to/models/sscd_disc_mixup.torchscript.pt
```

### Verify

```bash
export BRUSHNET_ROOT=/path/to/BrushNet   # defaults to ~/BrushNet if unset
test -f "$BRUSHNET_ROOT/src/diffusers/__init__.py" && echo "BrushNet source: OK"
test -f "$BRUSHNET_ROOT/checkpoints/stable-diffusion-xl-base-1.0/model_index.json" && echo "SDXL backbone: OK"
test -f "$BRUSHNET_ROOT/checkpoints/brushnet_segmentation_mask/config.json" && echo "BrushNet SD1.5 adapter: OK"
test -f "$BRUSHNET_ROOT/checkpoints/brushnet_sdxl/config.json" && echo "BrushNet SDXL adapter: OK"
```

## Run

There are four entry points: two interactive notebooks for fast
single-example iteration, and two long-running resumable evaluations.

### Interactive: Full Generation (SD1.5 or SDXL)

`notebooks/generation_comparison.ipynb` runs `clean_ddim`, `cads`, `tpso`,
and `rho_star` on one prompt. The `MODEL_FAMILY` toggle in its first cell
selects the backbone:

```python
MODEL_FAMILY = 'sd15'  # or 'sdxl'
```

For SDXL it points `MODEL_ID` at a local checkpoint — edit that line to
whichever checkpoint you downloaded above, e.g.:

```python
MODEL_ID = str(Path.home() / 'BrushNet' / 'checkpoints' / 'stable-diffusion-xl-base-1.0')
```

and raises the resolution to 1024x1024. All four methods are implemented and
verified for both families in this notebook.

### Interactive: BrushNet Inpainting (SD1.5 or SDXL)

`notebooks/inpainting_comparison.ipynb` runs `clean_ddim`, `cads`, `tpso`,
and `adaptive_pullback`/`rho_star_pullback` on one BrushBench example. Set
the same environment variables as the long inpainting run below (`BRUSHNET_
ROOT`, `BRUSHNET_MODEL_FAMILY`, `BRUSHBENCH_ROOT`, plus `BRUSHNET_BASE_MODEL`
if not using the code's default checkpoint folder name) before launching
Jupyter — the notebook's first cell reads them at import time, so exporting
them after the kernel has already started has no effect.

### Long Run: Full Generation (SD1.5 or SDXL)

Requires the COCO dataset and SSCD checkpoint from
[Datasets and metric checkpoints](#datasets-and-metric-checkpoints) above.

Edit `configs/generation.py`, especially:

- `MODEL_FAMILY` (`"sd15"` or `"sdxl"`) and `MODEL_ID` — for SDXL, point
  `MODEL_ID` at one of the checkpoints from
  [SDXL (full generation)](#sdxl-full-generation) above and raise `HEIGHT`/
  `WIDTH` to `1024`, e.g.:

  ```python
  MODEL_FAMILY = "sdxl"
  MODEL_ID = str(Path.home() / "BrushNet" / "checkpoints" / "stable-diffusion-xl-base-1.0")
  HEIGHT = 1024
  WIDTH = 1024
  ```

- `PROMPT_INDICES`
- `NUMBER_OF_PARTICLES`
- `PULLBACK_RANKS`
- `PULLBACK_POWER_ITERATIONS`
- `CANDIDATE_RHOS` — **not portable between families.** Rho scales the
  perturbation against each model's own token-embedding norms, and SD1.5 and
  SDXL sit on different scales. The SD1.5-tuned range is roughly
  `[0.10, 0.125, 0.15, 0.175, 0.20, 0.25]`; SDXL needs values around 10x
  larger, e.g. `[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]`. Re-tune rather than
  reuse the other family's values after switching `MODEL_FAMILY`.
- `RUN_NAME` — change this whenever `MODEL_FAMILY` changes, not only when a
  numeric parameter changes. `model_family` is part of the saved
  configuration identity, so resuming under the same `RUN_NAME` with a
  different family raises `RuntimeError: Configuration changed... Change
  RUN_NAME` instead of silently mixing SD1.5 and SDXL results.
- W&B settings

Validate without loading model weights:

```bash
python -m experiments.generation --check
```

Run the experiment:

```bash
python -u -m experiments.generation
```

The four method families are clean DDIM, CADS, TPSO, and one rho-star method
for every requested `(rank, power_iterations)` pair.

### Long Run: BrushNet Inpainting (SD1.5 or SDXL)

```bash
export BRUSHNET_ROOT=/path/to/BrushNet
export BRUSHNET_MODEL_FAMILY=sd15   # or sdxl
export BRUSHBENCH_ROOT=/path/to/BrushBench
```

By default, the SD1.5 base model is
`stable-diffusion-v1-5/stable-diffusion-v1-5` and its BrushNet checkpoint is
read from `$BRUSHNET_ROOT/checkpoints/brushnet_segmentation_mask`; the SDXL
base model defaults to `$BRUSHNET_ROOT/checkpoints/JuggernautXL-v9` and its
BrushNet checkpoint to `$BRUSHNET_ROOT/checkpoints/brushnet_sdxl`. Override
either with `BRUSHNET_BASE_MODEL` and `BRUSHNET_MODEL` when your checkpoint
folder names differ (e.g. the `stable-diffusion-xl-base-1.0` folder from
above needs `BRUSHNET_BASE_MODEL` set explicitly).

The current configuration in `configs/inpainting.py` compares `clean_ddim`,
`cads`, `tpso`, and `rho_star_pullback` on 500 BrushBench examples. Rho-star
is the only pullback variant in this comparison; the fixed-magnitude adaptive
variant is not run. It uses five independent initial particles, estimates
mask-aware pullback directions, and selects a nonzero perturbation magnitude
for each particle from `rho_star_candidate_rhos` (currently
`(1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)`, tuned for SDXL — see the rho-scale
note below). The rho-star probe maximizes pairwise DINO distance while
allowing at most a `0.35` decrease from each particle's clean CLIP score.

As with `CANDIDATE_RHOS` in full generation above, `rho_star_candidate_rhos`
and `pullback_rho` are **not portable between families**. The original
SD1.5-tuned range was `(0.10, 0.125, 0.15, 0.175, 0.20, 0.25)`; SDXL needs
values around 10x larger. Re-tune rather than reuse the other family's
values after switching `BRUSHNET_MODEL_FAMILY`.

Edit `RUN_NAME`, `MAX_EXAMPLES`, and the `rho_star_*` fields in that file to
start a different experiment. The comparison method list is:

```python
SELECTED_METHODS = ["rho_star_pullback", "clean_ddim", "cads", "tpso"]
```

Validate the configuration and local data:

```bash
python -m experiments.inpainting --check
```

Run the comparison:

```bash
python -u -m experiments.inpainting
```

Rebuild aggregate tables without loading the model:

```bash
python -m experiments.inpainting --aggregate-only
```

The current configuration uses independent initial particles and
`pullback_response_region="edit_mask"`. The response mask selects condition
directions that are locally expressive inside the editable region. Final
inpainting still restores the source pixels outside that region.

## Resume Behavior

Both long runners save each completed `(example, method)` result immediately.
To resume, keep the same configuration and `RUN_NAME`, then issue the same
command again. Completed work is restored and skipped.

If any scientific parameter changes, choose a new `RUN_NAME`. The runners stop
instead of mixing incompatible results in one output directory.

## Outputs

Generated artifacts are written under `outputs/` and ignored by Git. Each run
contains its resolved configuration, per-example images and JSON records,
per-example CSV rows, cumulative summaries, grids, timing, and method-specific
diagnostics. W&B is optional; local persistence does not depend on it.

## Verification

The fast CPU suites do not load model weights:

```bash
python -m tests.test_generation_math
python -m tests.test_inpainting_math
python -m tests.test_rho_star
```

They check:

- the shared DDIM transition and deterministic seeds;
- CADS fresh and fixed condition-noise streams;
- the known-spectrum block power and Rayleigh-Ritz basis;
- balanced fixed disjoint directions;
- the regional operator `J^T W_M^T W_M J`;
- pullback and CADS schedules;
- rho-star constraints and joint selection;
- independent inpainting initialization.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the migration audit,
metric definitions, seeds, and expected verification output.

## Important Scope Notes

- The pullback basis finds locally response-sensitive directions. It does not
  prove semantic disentanglement or global manifold preservation.
- The inpainting response mask selects directions; it does not spatially mask
  the prompt perturbation itself.
- The adaptive refresh basis is computed from one configured anchor particle
  and shared across particles. It is not recomputed independently for every
  particle.
- No model weights are trained or modified by these methods.
