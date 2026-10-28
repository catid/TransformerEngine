# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
"""Numerical tests for the reference CroissAttn JAX path."""

import jax
import jax.numpy as jnp
import numpy as np

from transformer_engine.jax.croissattn import (
    _attention_forward,
    _quantize_axis,
    croissattn_reference,
)


def test_croissattn_reference_finite_and_corrected_vjp():
    """The VJP must use O_aux in D*, rather than the rounded public output."""
    keys = jax.random.split(jax.random.key(17), 4)
    q, k, v = [jax.random.normal(x, (1, 16, 2, 16), dtype=jnp.bfloat16) for x in keys[:3]]
    doutput = jax.random.normal(keys[3], q.shape, dtype=jnp.bfloat16)

    output, saved = _attention_forward(q, k, v, 1.0 / 4.0, True)
    qh, kh, vh, p, pf, o_aux, groups = saved
    del groups
    assert output.dtype == q.dtype
    assert o_aux.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(output)))

    # Reproduce the CroissAttn backward equation directly from saved factors.
    doh = jnp.transpose(doutput, (0, 2, 1, 3)).astype(jnp.float32)
    # The JAX reference explicitly models consumer-axis recasts.  Native TE
    # uses independent SR streams; the portable path uses deterministic
    # nearest rounding, so reproduce that path in the expected equation.
    do_rec = _quantize_axis(doh, -1)
    v_rec = _quantize_axis(vh, -1)
    k_rec = _quantize_axis(kh, -2)
    dp = jnp.einsum("bhid,bhjd->bhij", do_rec, v_rec)
    dstar = jnp.sum(doh * o_aux, axis=-1, keepdims=True)
    ds = p * (dp - dstar)
    expected_q = jnp.transpose(
        jnp.einsum("bhij,bhjd->bhid", ds, k_rec) * 0.25,
        (0, 2, 1, 3),
    ).astype(q.dtype)
    actual = jax.grad(
        lambda z: jnp.sum(
            croissattn_reference(z, k, v, 0.25, True).astype(jnp.float32)
            * doutput.astype(jnp.float32)
        )
    )(q)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected_q), rtol=2e-2, atol=2e-2)


def test_croissattn_reference_gqa_jit_vjp():
    """GQA backward must trace through the tiled replay under ``jax.jit``.

    The replay loop uses dynamic tile offsets.  A Python slice whose bounds
    depend on the loop index raises during tracing, so exercise the same JIT
    path used by MaxText and check the collapsed KV-head gradients.
    """
    keys = jax.random.split(jax.random.key(23), 3)
    q = jax.random.normal(keys[0], (1, 32, 4, 16), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (1, 32, 2, 16), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (1, 32, 2, 16), dtype=jnp.bfloat16)

    def loss(q_, k_, v_):
        return jnp.sum(croissattn_reference(q_, k_, v_, 0.25, True).astype(jnp.float32))

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)
    assert [g.shape for g in grads] == [q.shape, k.shape, v.shape]
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

