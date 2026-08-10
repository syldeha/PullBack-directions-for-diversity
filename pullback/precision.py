"""Numerical-safety helpers shared by generation and inpainting."""

from contextlib import contextmanager

import torch


def module_dtype(module):
    """Return a module's execution dtype across Diffusers/PyTorch classes."""

    dtype = getattr(module, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(module.parameters()).dtype
    except StopIteration as error:
        raise TypeError("cannot infer dtype from a parameterless module") from error


def require_finite(name, tensor):
    """Raise at the first non-finite model intermediate.

    Converting a NaN image to uint8 can look like a valid black image.  The
    explicit check keeps that failure close to the VAE operation that caused
    it and gives the user an actionable error instead.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return tensor
    bad = int((~finite).sum().item())
    total = tensor.numel()
    raise FloatingPointError(
        f"{name} contains {bad}/{total} NaN or Inf values "
        f"(dtype={tensor.dtype}, device={tensor.device})"
    )


def vae_requires_upcast(vae):
    """Return whether a half-precision VAE declares fp32 execution."""

    return bool(
        module_dtype(vae) == torch.float16
        and getattr(vae.config, "force_upcast", False)
    )


@contextmanager
def vae_precision(vae):
    """Temporarily move a force-upcast VAE to fp32, then restore its dtype."""

    original_dtype = module_dtype(vae)
    upcast = vae_requires_upcast(vae)
    if upcast:
        vae.to(dtype=torch.float32)
    try:
        yield torch.float32 if upcast else original_dtype
    finally:
        if upcast and module_dtype(vae) != original_dtype:
            vae.to(dtype=original_dtype)
