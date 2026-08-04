# Executed Methodology

This document describes the implementation in this repository. It separates
the shared mathematical method from the two model-specific pipelines.

## 1. Shared Reverse Process

For particle `i`, classifier-free guidance uses

```math
\epsilon_{\mathrm{CFG}}^{(i)}
=
\epsilon_\theta(z_t^{(i)},t,C_t^-)
+w\left[
\epsilon_\theta(z_t^{(i)},t,C_{i,t}^+)
-\epsilon_\theta(z_t^{(i)},t,C_t^-)
\right].
```

The scheduler first scales the U-Net input. The DDIM update then receives the
original unscaled latent and the CFG prediction. Every compared method receives
the exact same initial latent tensor and DDIM eta-noise seed.

The default experiments use `eta=0`, so no posterior noise is injected during
the DDIM trajectory.

## 2. Conditional Pullback Operator

At a fixed latent, timestep, and model context, define the positive conditional
response

```math
f(C)=\epsilon_\theta(z_t,t,C),
\qquad
J_C=\frac{\partial f(C)}{\partial C}.
```

The global pullback Gram operator is

```math
G_C=J_C^T J_C.
```

For a condition direction `v`, the implementation computes only `G_C v`.
It never stores `J_C` or `G_C`.

### 2.1 Matrix-Free Product

First, a centered finite difference approximates the JVP:

```math
J_Cv
\approx
\frac{f(C+\delta v)-f(C-\delta v)}{2\delta}.
```

Then autograd computes the VJP:

```math
J_C^T(J_Cv)
=
\nabla_C\langle f(C),\operatorname{stopgrad}(J_Cv)\rangle.
```

The finite-difference response is detached. Therefore this is one first-order
VJP, not a second-order differentiation through the finite-difference calls.

### 2.2 Regional Inpainting Operator

Let `W_M` multiply a denoiser response by the binary edit mask resized to
latent resolution and broadcast across response channels. Regional direction
discovery uses

```math
G_{C,M}v=J_C^T W_M^T W_M J_Cv.
```

Both the differentiable base response and the finite-difference JVP are masked
before the VJP. For a binary mask, `W_M^T W_M = W_M`.

At fixed `z_t` and `t`, the predicted clean latent is

```math
\widehat z_0(C)
=
\frac{z_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t,t,C)}
{\sqrt{\bar\alpha_t}}.
```

Its condition Jacobian differs from the epsilon Jacobian by a scalar. Thus the
masked epsilon-space and masked clean-latent operators have the same
eigenvectors. This statement does not extend through the nonlinear VAE decoder
to exact RGB geometry.

## 3. Dominant Basis

The requested rank `K` is a hyperparameter. The code draws a Gaussian block
`Q_0` in the real-token condition space and orthonormalizes it with QR.

For `r = 0, ..., R-1`:

```math
Z_{r+1}=GQ_r,
\qquad
Q_{r+1}=\operatorname{QR}(Z_{r+1}).
```

This block power iteration approaches the dominant `K`-dimensional eigenspace.
QR prevents all block columns from collapsing onto only the leading
eigenvector.

The resulting orthonormal block identifies a subspace, but its vectors may be
rotated mixtures of eigenvectors. The code therefore forms the small Rayleigh
matrix

```math
H=Q^T GQ,
```

symmetrizes it, diagonalizes it, sorts its eigenvalues in descending order, and
lifts the small eigenvectors back to condition space. This is the
Rayleigh-Ritz step. The output basis contains approximate dominant right
singular directions of `J_C` because the eigenvectors of `J_C^T J_C` are the
right singular vectors of `J_C`.

The rank is reduced only if numerical QR detects a dependent direction.

## 4. Fixed Disjoint Directions

For `N` particles and rank `K`, basis indexes are distributed with a balanced
snake assignment. For example, rank 11 and 4 particles gives:

```text
particle 0: [0, 7, 8]
particle 1: [1, 6, 9]
particle 2: [2, 5, 10]
particle 3: [3, 4]
```

Each particle owns a disjoint basis subset. One persistent ambient Gaussian
field is sampled per particle. It is projected onto that particle's current
subset and normalized. When an adaptive basis is refreshed, the same ambient
field is projected again. This is the implemented meaning of fixed noise:
particle identity persists across timesteps and basis refreshes.

The direction is scaled relative to the clean prompt:

```math
\Delta C_i=\rho_i S_Cd_i,
```

where `d_i` has unit global norm and `S_C` multiplies each real token by the
norm of its clean token embedding. Padding tokens are not perturbed.

## 5. Scheduled Prompt Perturbation

The pullback envelope is

```math
\alpha(t)=
\begin{cases}
1, & t\geq T_S,\\
\left(\dfrac{t-T_E}{T_S-T_E}\right)^p,
   & T_E<t<T_S,\\
0, & t\leq T_E.
\end{cases}
```

The sampler uses

```math
C_{i,t}^+=C_0^++\alpha(t)\rho_iS_Cd_{i,t},
\qquad C_{i,t}^-=C_0^-.
```

The clean negative branch is retained for pullback and TPSO. CADS is the
intentional exception: it noises and rescales both branches.

## 6. Adaptive Basis Refresh

The initial basis is computed from a clean prefix of one anchor particle at the
configured basis timestep. The actual batch sampler then restarts from the
original untouched initial latents.

At configured refresh positions inside `(T_E, T_S)`, a low-rank basis is
recomputed from one configured anchor particle's current latent. It is shared
by all particles. The old and refreshed particle directions are linearly
interpolated and renormalized during a short transition window.

This is a practical shared local approximation. It is not parallel transport,
and it does not estimate one Jacobian per particle.

## 7. Particle-Specific Rho Selection

The full-generation rho-star method first computes a clean prefix from each
original particle to timestep `t_p`. Candidate scales include `rho=0` as a
clean reference, but the configured positive candidate list determines which
nonzero values may be selected.

For candidate `m` of particle `i`, the probe condition is

```math
C_{i,m}^{\mathrm{probe}}
=C_0^++\alpha(t_p)\rho_mS_Cd_i.
```

The code predicts a CFG Tweedie clean latent at `t_p`, decodes that estimate,
and computes its CLIP score and normalized DINO feature. It does not finish a
complete trajectory for every candidate.

The selected assignment maximizes average pairwise DINO cosine distance:

```math
\max_{m_1,\ldots,m_N}
\frac{2}{N(N-1)}
\sum_{i<j}
\left(1-h_{i,m_i}^T h_{j,m_j}\right),
```

subject to the one-sided alignment constraint

```math
s_{\mathrm{CLIP}}(\widehat x_{i,m_i},C)
\geq
s_{\mathrm{CLIP}}(\widehat x_{i,0},C)-\delta_{\mathrm{CLIP}}.
```

An improvement over the clean CLIP score is always allowed. Ties favor the
smaller total rho. Exact enumeration is used when requested and feasible;
otherwise beam search approximates the same objective. If no selectable
candidate satisfies the constraint, the configured fallback is recorded.

After selection, the probe is discarded and sampling restarts from the exact
original initial latent tensor.

## 8. CADS

CADS uses

```math
\gamma(t)=
\begin{cases}
0, & t\geq T_S,\\
1, & t\leq T_E,\\
\dfrac{T_S-t}{T_S-T_E}, & T_E<t<T_S.
\end{cases}
```

For each CFG branch `b`:

```math
\widehat C_t^b
=
\sqrt{\gamma(t)}C^b
+s\sqrt{1-\gamma(t)}\,\xi_t^b.
```

Positive and negative noise are independent. Each noised branch is rescaled
to the mean and standard deviation of its corresponding clean branch before
CFG. The common experiment configuration uses fresh noise and full rescaling.

## 9. TPSO

TPSO learns additive content-token offsets with the text encoder and diffusion
model frozen. Its semantic term constrains each optimized pooled text feature
relative to the clean prompt feature. Its diversity term minimizes mean
off-diagonal cosine similarity across particle prompt features.

During sampling, the optimized positive conditions linearly return to the
clean condition over the configured early fraction of the reverse process.
The negative condition stays clean.

## 10. Model-Specific Differences

### Stable Diffusion 1.5

- The response map is the positive U-Net epsilon prediction.
- The condition is the CLIP token hidden state for real tokens.
- Rho-star is implemented in this track.
- The long comparison uses clean DDIM, CADS, TPSO, and rho-star adaptive
  pullback.

### BrushNet SDXL Inpainting

- Source image, binary mask, pooled SDXL condition, and BrushNet residual path
  are fixed while differentiating the positive token condition.
- The differentiated response is the positive conditional epsilon prediction,
  not CFG and not decoded RGB.
- The long comparison uses a fixed pullback rho, not rho-star.
- The final blended outputs restore source pixels outside the edit mask.
- `response_region="global"` reproduces the completed long evaluation.
- `response_region="edit_mask"` enables the later regional operator and must
  be treated as a distinct experiment.

