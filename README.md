# Pullback-Guided Diffusion

This repository contains the cleaned research implementation of conditional
pullback sampling for diffusion models. It supports two experiment tracks:

1. Full image generation with Stable Diffusion 1.5.
2. Image inpainting with BrushNet and an SDXL checkpoint.

The code compares clean DDIM, CADS, TPSO, and pullback-guided sampling. The
full-generation track additionally implements particle-specific rho selection
with a short CLIP-constrained DINO probe.

This repository contains code only.

## Repository Layout

```text
pullback-diffusion/
|-- configs/       # Parameters edited before an experiment
|-- generation/    # SD1.5 loading, prompt encoding, CFG, and DDIM
|-- pullback/      # Matrix-free pullback basis and disjoint directions
|-- methods/       # CADS, TPSO, adaptive pullback, and rho-star for SD1.5
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
linearly to the clean condition during sampling.

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

## Installation

The code was verified with Python 3.10. Install the common dependencies with:

```bash
python -m pip install -r requirements.txt
```

For full generation, make sure the configured Stable Diffusion 1.5 checkpoint
is available in the Hugging Face cache, or set `LOCAL_FILES_ONLY = False` for
the first download.

BrushNet inpainting uses the custom Diffusers source shipped by the upstream
BrushNet repository. Clone and install it separately:

```bash
git clone https://github.com/TencentARC/BrushNet.git
cd BrushNet
python -m pip install -e .
```

The inpainting runner supports SD1.5 and SDXL. The current BrushBench
rho-star experiment uses Stable Diffusion 1.5 with the BrushNet segmentation
mask checkpoint. Point this repository to the external BrushNet checkout and
downloaded BrushBench data with:

```bash
export BRUSHNET_ROOT=/path/to/BrushNet
export BRUSHNET_MODEL_FAMILY=sd15
export BRUSHBENCH_ROOT=/path/to/BrushBench
```

By default, the SD1.5 base model is
`stable-diffusion-v1-5/stable-diffusion-v1-5` and the BrushNet checkpoint is
read from `$BRUSHNET_ROOT/checkpoints/brushnet_segmentation_mask`. Override
them with `BRUSHNET_BASE_MODEL` and `BRUSHNET_MODEL` when needed.

For the local setup used in this repository:

```bash
export BRUSHNET_ROOT="/home/dehay/Sylvain/Image_inpaiting pullback/BrushNet"
export BRUSHBENCH_ROOT="/home/dehay/Sylvain/pullback-diffusion/download/BrushnetDataEval"
export BRUSHNET_MODEL_FAMILY=sd15
```

Follow the upstream BrushNet data and checkpoint licenses. They are not
redistributed here.

## Full Generation

### Prepare COCO 2017

Choose the protocol in `configs/coco2017.py`, then run:

```bash
export COCO_ROOT=/path/to/coco2017
python -m scripts.prepare_coco2017
```

The script downloads the official validation images and annotations when they
are missing, then builds the seeded caption manifest.

The full-generation metrics also require the SSCD TorchScript checkpoint:

```bash
export SSCD_MODEL_PATH=/path/to/sscd_disc_mixup.torchscript.pt
```

### Configure And Run

Edit `configs/generation.py`, especially:

- `PROMPT_INDICES`
- `NUMBER_OF_PARTICLES`
- `PULLBACK_RANKS`
- `PULLBACK_POWER_ITERATIONS`
- `CANDIDATE_RHOS`
- `RUN_NAME`
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

## BrushNet Inpainting

The current configuration in `configs/inpainting.py` compares `clean_ddim`,
`cads`, `tpso`, and `rho_star_pullback` on 500 BrushBench examples. Rho-star
is the only pullback variant in this comparison; the fixed-magnitude adaptive
variant is not run. It uses five independent initial particles, estimates
mask-aware pullback directions, and selects a nonzero perturbation magnitude
for each particle from
`{0.10, 0.125, 0.15, 0.175, 0.20, 0.25}`. The rho-star probe maximizes
pairwise DINO distance while allowing at most a `0.35` decrease from each
particle's clean CLIP score.

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
