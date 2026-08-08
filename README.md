# Pullback-Guided Diffusion

Research code for conditional pullback sampling on diffusion models. Two experiment tracks live here:

1. Full image generation, Stable Diffusion 1.5 or SDXL.
2. Image inpainting with BrushNet, also SD1.5 or SDXL.

Every track compares clean DDIM, CADS, TPSO, and pullback-guided sampling. Full generation also runs a particle-specific rho selection step — a short CLIP-constrained DINO probe.

Code only — no trained weights ship with this repo.

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

The scientific code stays separate from the evaluation machinery on purpose. The pullback basis itself lives in `pullback/basis.py`; `experiments/generation.py` just wires together captions, checkpoints, metrics, tables, and W&B logging.

## Implemented Methods

### Clean DDIM

The baseline. Independent initial Gaussian latents, unmodified positive/negative conditions at every CFG-DDIM step.

### CADS

Follows the original condition-annealing setup used throughout this project:

- isotropic Gaussian noise on both CFG branches
- positive and negative condition noise drawn independently
- each branch rescaled against its own clean condition
- fresh condition noise every active step (the public config)
- every method still gets the exact same initial latent tensor

### TPSO

Learns one positive content-token offset per particle. BOS, EOS, and padding stay fixed; the text encoder and diffusion model stay frozen. The negative branch stays clean, and the optimized condition eases back to clean over the course of sampling. On SDXL, offsets get optimized jointly on both text encoders, then their penultimate hidden states get concatenated — same as `StableDiffusionXLPipeline.encode_prompt`.

### Adaptive Pullback

Estimates the dominant directions of the condition-response Gram operator without ever building a full Jacobian:

```math
G_C v = J_C^T J_C v.
```

For regional inpainting direction discovery, swap in:

```math
G_{C,M}v = J_C^T W_M^T W_M J_Cv.
```

Centered finite differences give `Jv`, autograd handles the VJP, and block power iteration with Rayleigh-Ritz extraction pulls out the requested basis. Fixed Gaussian fields get projected into balanced disjoint subsets, so each particle keeps its identity across basis refreshes.

### Rho-Star

Picks one perturbation magnitude per particle for full generation. At a single intermediate timestep, it decodes only the Tweedie candidate estimates, throws out anything whose CLIP score drops too far, and jointly maximizes mean pairwise DINO distance. Then it restarts the trajectory from the original latent with whatever scales won.

Full equations and the exact choices made are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Install

You'll need Python 3.10. Install the shared dependencies:

```bash
python -m pip install -r requirements.txt
```

BrushNet inpainting — both SD1.5 and SDXL — needs the custom Diffusers source from upstream BrushNet. Clone it and install separately:

```bash
git clone https://github.com/TencentARC/BrushNet.git ~/BrushNet
cd ~/BrushNet
python -m pip install -e .
```

`BRUSHNET_ROOT` defaults to `~/BrushNet` everywhere below. Only export it if you cloned somewhere else:

```bash
export BRUSHNET_ROOT=/path/to/BrushNet
```

Follow upstream BrushNet's data and checkpoint licenses — nothing from them is redistributed here.

## Set Up Checkpoints

Every checkpoint goes at the exact local path the code already defaults to. Once it's downloaded, there's nothing else to configure unless a section below says otherwise.

### Stable Diffusion 1.5 (full generation)

Nothing to do here. `stable-diffusion-v1-5/stable-diffusion-v1-5` downloads itself into the Hugging Face cache the first time `experiments.generation` or `notebooks/generation_comparison.ipynb` runs.

### SDXL (full generation)

Full generation and BrushNet inpainting share the same SDXL backbone format, so you only need to download this once:

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

Notice the patterns spell out exact fp16 filenames instead of a `*.safetensors` wildcard. That's deliberate — this repo hosts both fp32 and fp16 weights, and a broad wildcard pulls down both (~20 GB instead of ~6.5 GB) even though only the fp16 files ever get loaded.

One more thing worth knowing: this VAE overflows to NaN (solid black output) when run in plain fp16, in either direction -- decoding latents into pixels or encoding pixels into latents. It flags that itself via `force_upcast: true` in its config. `generation/model.py` and `inpainting/model.py`'s `decode_latents` already check for that flag and decode in fp32 when needed, and `inpainting/model.py`'s `encode_image_latent` -- used to build BrushNet's masked-image conditioning latent -- does the same on the encode side, so there's nothing extra to configure. Just know it's why that logic is there, and why both directions need the same guard: a NaN slipping in on the encode side poisons every subsequent denoising step just as surely as an unguarded decode does.

Prefer a community checkpoint instead? [RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) works the same way — same download pattern, just swap `repo_id="RunDiffusion/Juggernaut-XL-v9"` and `local_dir=".../checkpoints/JuggernautXL-v9"`. That folder name happens to match `inpainting/model.py`'s built-in default for `BRUSHNET_MODEL_FAMILY=sdxl`, so you can skip the `BRUSHNET_BASE_MODEL` override in that one case. Any other folder name — including `stable-diffusion-xl-base-1.0` above — needs it.

### BrushNet adapter — SD1.5 (inpainting)

The SD1.5 BrushNet checkpoint (`segmentation_mask_brushnet_ckpt`) has no Hugging Face mirror. Like the SDXL adapter below, it only lives inside one shared Google Drive folder alongside 10 unrelated checkpoints (https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n). Grab just the two files you need, by ID, straight into the code's default path:

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

Those IDs came straight from the folder listing (see the SDXL adapter section below for how to re-list it), but weren't independently downloaded and verified in this pass — if the download fails, re-run the listing command to confirm them.

### BrushNet adapter — SDXL (inpainting)

Same Google Drive folder, this time for `segmentation_mask_brushnet_ckpt_sdxl_v1`. Downloading the whole folder isn't practical — 11 unrelated checkpoints in there — so list its contents first without downloading anything, just to get the file IDs:

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

Then pull just those two files by ID into the default adapter path:

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

As of this writing, `config.json` is `1jCt6KsP2jKSBD_QYJsuQ9Yb1rujLLt3f` and `diffusion_pytorch_model.safetensors` is `1yPhfQ3y60fXURr9_YfQrdE48y68IZjBN` — both downloaded and verified working in this repo. Drive file IDs can shift if the folder gets reorganized, though, so re-run the listing above if either one fails.

### Datasets and metric checkpoints

**BrushBench** (inpainting evaluation data):

```bash
export BRUSHBENCH_ROOT=/path/to/BrushBench
```

Get the data by following the upstream BrushNet instructions — it isn't redistributed here.

**COCO 2017** (full-generation evaluation data). Pick the protocol in `configs/coco2017.py`, then run:

```bash
export COCO_ROOT=/path/to/coco2017
python -m scripts.prepare_coco2017
```

It downloads the official validation images and annotations if they're missing, then builds the seeded caption manifest.

**SSCD** (full-generation metrics) — Meta's official release for image copy detection (`facebookresearch/sscd-copy-detection`):

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

Four entry points: two interactive notebooks for fast single-example iteration, and two long-running resumable evaluations.

### Interactive: Full Generation (SD1.5 or SDXL)

`notebooks/generation_comparison.ipynb` runs `clean_ddim`, `cads`, `tpso`, and `rho_star` on one prompt. Pick the backbone with the `MODEL_FAMILY` toggle in its first cell:

```python
MODEL_FAMILY = 'sd15'  # or 'sdxl'
```

For SDXL, point `MODEL_ID` at whichever checkpoint you downloaded above, e.g.:

```python
MODEL_ID = str(Path.home() / 'BrushNet' / 'checkpoints' / 'stable-diffusion-xl-base-1.0')
```

That also bumps the resolution to 1024x1024. All four methods are implemented and verified for both families here.

### Interactive: BrushNet Inpainting (SD1.5 or SDXL)

`notebooks/inpainting_comparison.ipynb` runs `clean_ddim`, `cads`, `tpso`, and `adaptive_pullback`/`rho_star_pullback` on one BrushBench example. Set the same environment variables as the long inpainting run below (`BRUSHNET_ROOT`, `BRUSHNET_MODEL_FAMILY`, `BRUSHBENCH_ROOT`, plus `BRUSHNET_BASE_MODEL` if you're not using the code's default checkpoint folder name) before you launch Jupyter. The notebook's first cell reads them at import time, so exporting them after the kernel's already running won't do anything.

### Long Run: Full Generation (SD1.5 or SDXL)

Needs the COCO dataset and SSCD checkpoint from [Datasets and metric checkpoints](#datasets-and-metric-checkpoints) above.

Edit `configs/generation.py`, especially:

- `MODEL_FAMILY` (`"sd15"` or `"sdxl"`) and `MODEL_ID` — for SDXL, point `MODEL_ID` at one of the checkpoints from [SDXL (full generation)](#sdxl-full-generation) above and bump `HEIGHT`/`WIDTH` to `1024`:

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
- `CANDIDATE_RHOS` — **doesn't carry over between families.** Rho scales the perturbation against each model's own token-embedding norms, and SD1.5 and SDXL just sit on different scales. SD1.5's tuned range is roughly `[0.10, 0.125, 0.15, 0.175, 0.20, 0.25]`; SDXL wants values around 10x bigger, something like `[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]`. Re-tune it rather than reuse the other family's numbers after flipping `MODEL_FAMILY`.
- `RUN_NAME` — change this whenever `MODEL_FAMILY` changes, not just for numeric tweaks. `model_family` is baked into the saved configuration identity now, so resuming under the same `RUN_NAME` with a different family throws `RuntimeError: Configuration changed... Change RUN_NAME` instead of quietly mixing SD1.5 and SDXL results together.
- W&B settings

Validate without loading model weights:

```bash
python -m experiments.generation --check
```

Then run it for real:

```bash
python -u -m experiments.generation
```

Four method families: clean DDIM, CADS, TPSO, and one rho-star method per requested `(rank, power_iterations)` pair.

### Long Run: BrushNet Inpainting (SD1.5 or SDXL)

```bash
export BRUSHNET_ROOT=/path/to/BrushNet
export BRUSHNET_MODEL_FAMILY=sd15   # or sdxl
export BRUSHBENCH_ROOT=/path/to/BrushBench
```

By default, SD1.5 uses `stable-diffusion-v1-5/stable-diffusion-v1-5` as its base model and reads its BrushNet checkpoint from `$BRUSHNET_ROOT/checkpoints/brushnet_segmentation_mask`. SDXL defaults to `$BRUSHNET_ROOT/checkpoints/JuggernautXL-v9` for the base model and `$BRUSHNET_ROOT/checkpoints/brushnet_sdxl` for its BrushNet checkpoint. If your checkpoint folder names differ — say, the `stable-diffusion-xl-base-1.0` folder from above — override with `BRUSHNET_BASE_MODEL` and `BRUSHNET_MODEL`.

The current setup in `configs/inpainting.py` compares `clean_ddim`, `cads`, `tpso`, and `rho_star_pullback` on 500 BrushBench examples. Rho-star is the only pullback variant here — the fixed-magnitude adaptive version doesn't run. It uses five independent initial particles, estimates mask-aware pullback directions, and picks a nonzero perturbation magnitude per particle from `rho_star_candidate_rhos` (currently `(1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)`, tuned for SDXL — see the note below). The rho-star probe maximizes pairwise DINO distance while allowing at most a `0.35` drop in each particle's clean CLIP score.

Same caveat as `CANDIDATE_RHOS` above: `rho_star_candidate_rhos` and `pullback_rho` don't carry over between families either. The original SD1.5-tuned range was `(0.10, 0.125, 0.15, 0.175, 0.20, 0.25)`; SDXL wants roughly 10x that. Re-tune after switching `BRUSHNET_MODEL_FAMILY` rather than reusing the other family's numbers.

Edit `RUN_NAME`, `MAX_EXAMPLES`, and the `rho_star_*` fields in that file to start a different run. The comparison list:

```python
SELECTED_METHODS = ["rho_star_pullback", "clean_ddim", "cads", "tpso"]
```

Validate the configuration and local data:

```bash
python -m experiments.inpainting --check
```

Run it:

```bash
python -u -m experiments.inpainting
```

Rebuild aggregate tables without touching the model:

```bash
python -m experiments.inpainting --aggregate-only
```

This configuration uses independent initial particles and `pullback_response_region="edit_mask"`. The response mask picks out condition directions that are locally expressive inside the editable region — final inpainting still restores the source pixels everywhere outside it.

## Resume Behavior

Both long runners save each completed `(example, method)` result immediately. To resume, keep the same configuration and `RUN_NAME` and just run the same command again — completed work gets restored and skipped.

Changed a scientific parameter? Pick a new `RUN_NAME`. The runners stop rather than mix incompatible results into one output directory.

## Outputs

Generated artifacts land under `outputs/`, ignored by Git. Each run keeps its resolved configuration, per-example images and JSON records, per-example CSV rows, cumulative summaries, grids, timing, and method-specific diagnostics. W&B is optional — local persistence doesn't depend on it.

## Verification

The fast CPU suites don't load model weights:

```bash
python -m tests.test_generation_math
python -m tests.test_inpainting_math
python -m tests.test_rho_star
```

What they check:

- the shared DDIM transition and deterministic seeds
- CADS fresh and fixed condition-noise streams
- the known-spectrum block power and Rayleigh-Ritz basis
- balanced fixed disjoint directions
- the regional operator `J^T W_M^T W_M J`
- pullback and CADS schedules
- rho-star constraints and joint selection
- independent inpainting initialization

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the migration audit, metric definitions, seeds, and expected verification output.

## Important Scope Notes

- The pullback basis finds locally response-sensitive directions. That's not proof of semantic disentanglement or global manifold preservation.
- The inpainting response mask selects directions — it doesn't spatially mask the prompt perturbation itself.
- The adaptive refresh basis comes from one configured anchor particle, shared across particles. It isn't recomputed independently for every particle.
- No model weights get trained or modified by any of this.
