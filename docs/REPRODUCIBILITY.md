# Reproducibility

## 1. Fair Comparison Contract

Within one prompt or BrushBench example, every method receives:

- the same model checkpoint;
- the same prompt and negative prompt;
- the same scheduler and number of reverse steps;
- the same CFG scale and eta;
- the exact same initial latent tensor;
- the same eta-noise seed;
- the same source image, mask, and BrushNet condition for inpainting.

Method-specific randomness has separate deterministic seeds. Reusing only the
same distribution is not considered sufficient; the runner records or reuses
the same tensors.

## 2. Canonical Configurations

### Full generation

Edit `configs/generation.py`. Its default scientific settings are the final
five-particle SD1.5 comparison:

- 50 DDIM steps, CFG 7.5, eta 0;
- independent initial latent batch from seed 12345 plus prompt index;
- rank 40, two block-power iterations;
- pullback basis at timestep 500;
- two rank-20 adaptive refreshes;
- fixed disjoint prompt noise;
- quadratic schedule from 999 to 500;
- rho candidates `{0.10, 0.125, 0.15, 0.175, 0.20, 0.25}`;
- probe timestep 699;
- one-sided CLIP-drop tolerance 0.35;
- CADS fresh noise, scale 0.15, window 900 to 600, full rescaling.

The selected caption count is controlled independently by `PROMPT_INDICES`.

### BrushNet inpainting

Edit `configs/inpainting.py`. The audited long-run settings are:

- 4 independent particles, initial seed 4242;
- resolution 1024;
- 50 DDIM steps, eta 0;
- CADS scale 0.15, window 900 to 600, fresh isotropic branch noise;
- pullback rank 24, two block-power iterations, basis timestep 600;
- fixed disjoint pullback rho 1.25;
- quadratic schedule from 999 to 500;
- two rank-8, one-iteration adaptive refreshes;
- two-step direction interpolation;
- global response operator;
- TPSO parameters shown directly in the configuration file.

## 3. Resume And Checkpoint Rules

Both runners write one result per `(example, method)` before continuing. They
also save a complete resolved configuration.

To resume:

1. Keep the same configuration and `RUN_NAME`.
2. Run the same command again.
3. The runner verifies the configuration, restores complete records, and skips
   them.

To change any scientific parameter, choose a new `RUN_NAME`. This prevents
mixed results.

Generation:

```bash
python -u -m experiments.generation
```

Inpainting:

```bash
python -u -m experiments.inpainting
```

Rebuild inpainting tables only:

```bash
python -m experiments.inpainting --aggregate-only
```

## 4. Metric Definitions

### CLIP

CLIP uses normalized image and text features from
`openai/clip-vit-base-patch32`. Reported CLIPScore is 100 times cosine
similarity. Higher is better.

### DINO

DINO uses normalized `vit_base_patch14_dinov2` image features from `timm`.
`dino_sim_mean` is the mean off-diagonal pairwise cosine similarity and
`dino_sim_max` is the largest pairwise similarity. Lower is more diverse.

### LPIPS

The inpainting comparison uses LPIPS with the AlexNet backbone on padded crops
around the edit mask. `lpips_mean` and `lpips_min` summarize all image pairs.
Higher is more diverse.

### MSS And Vendi

The COCO comparison follows the CADS-style SSCD protocol. SSCD features are
normalized before constructing the similarity matrix. MSS includes diagonal
self-similarity in the generation evaluator, matching the historical run.
Vendi is the exponential entropy of the normalized kernel eigenvalues. Lower
MSS and higher Vendi indicate more diversity.

The inpainting evaluator historically computes MSS and Vendi from normalized
CLIP image features. This difference is explicit and should not be hidden when
comparing numbers across the two experiment tracks.

### Mask-Crop Metrics

The inpainting runner also computes CLIP, DINO, MSS, and Vendi on a padded crop
around the edit mask. Raw outside-mask MAE and PSNR are optional diagnostics;
final blended outputs exactly restore the source outside the mask.

## 5. Known Limits

- CUDA kernels can introduce small cross-machine numerical differences even
  with fixed seeds.
- The basis uses a fixed number of subspace iterations and no residual-based
  convergence test.
- The adaptive basis is estimated from one anchor particle and shared.
- The rho-star probe uses intermediate Tweedie estimates, not final images for
  every candidate.
- The completed BrushBench run used the global operator. Regional pullback is
  implemented but represents a separate experiment.
- The two evaluators use different MSS/Vendi feature backbones, so their
  absolute values are not directly comparable.
