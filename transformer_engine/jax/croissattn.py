"""Reference CroissAttn attention for JAX.

This module is intentionally a reference implementation.  It follows the
CroissAttn dataflow (NVFP4 Q/K/V factors, an unrounded ``O_aux`` saved for the
softmax derivative, and the corrected ``D* = <dO, O_aux>`` term), but uses JAX
matrix multiplies for the attention contraction.  It is useful for numerical
and integration tests while a fused SM100 custom call is being developed.

The residual and backward traversal mirror the complete contract: Q/K/V/U
payloads and row statistics are replayed in fixed tiles, ``O_aux`` is
recomputed tile by tile, and each consumer contraction gets an explicit
axis recast.  This backend uses deterministic nearest rounding and ordinary
JAX FP32 accumulators; it does not provide the native kernel's independent
stochastic-rounding streams or packed NVFP4 GEMMs.  Throughput numbers from
this path therefore measure integration correctness, not production TE speed.

The function is separate from the cuDNN fused-attention primitive because that
primitive accepts BF16/FP16 tensors and does not expose an NVFP4 factor pack or
an auxiliary output to its JAX backward rule.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


# Keep host constants as Python values.  A module-level ``jnp.asarray`` leaks
# a device array into MaxText's remat/checkpoint transform when this module is
# imported inside the traced layer function (JAX raises UnexpectedTracerError).
_E2M1_LEVELS_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
_BLOCK = 16
_LOG2E = math.log2(math.e)
_REPLAY_TILE = 128


def _e4m3_decode(code: jax.Array) -> jax.Array:
    """Decode positive finite E4M3FN codes without a float8 dependency."""
    code = code.astype(jnp.int32) & 0x7F
    exp = code >> 3
    mant = code & 7
    normal = jnp.ldexp((8 + mant).astype(jnp.float32), exp - 10)
    subnormal = jnp.ldexp(mant.astype(jnp.float32), jnp.full_like(exp, -9))
    return jnp.where(exp > 0, normal, subnormal)


def _e4m3_ceil(x: jax.Array) -> jax.Array:
    """Smallest positive E4M3FN value covering ``x`` (x >= 0)."""
    # A small lookup is slower than hardware float8 conversion but keeps the
    # reference independent of the JAX float8 implementation and exact at the
    # subnormal boundary.
    codes = jnp.arange(1, 127, dtype=jnp.int32)
    values = _e4m3_decode(codes)
    idx = jnp.sum(values < jnp.clip(x[..., None], 0.0, _E4M3_MAX), axis=-1)
    return values[jnp.clip(idx, 0, 125)]


def _e2m1_nearest(x: jax.Array) -> jax.Array:
    """Round to nearest E2M1 value with a saturating magnitude."""
    levels = jnp.asarray(_E2M1_LEVELS_VALUES, dtype=jnp.float32)
    sign = jnp.signbit(x)
    a = jnp.minimum(jnp.abs(x), _E2M1_MAX)
    # Compare against midpoints.  ``argmin`` provides the same tie behavior
    # for this reference path as the nearest conversion used by CroissAttn.
    dist = jnp.abs(a[..., None] - levels)
    y = levels[jnp.argmin(dist, axis=-1)]
    return jnp.where(sign, -y, y)


def _quantize_last_axis(x: jax.Array, block: int = _BLOCK) -> jax.Array:
    """Decode an NVFP4 block-quantized tensor whose blocks are on the last axis."""
    x = x.astype(jnp.float32)
    if x.shape[-1] % block:
        raise ValueError(f"last dimension {x.shape[-1]} must be divisible by {block}")
    # CroissAttn uses one covering tensor multiplier per Q/K/V factor.  The
    # E4M3 block scales below then provide the local 16-element dynamic range.
    amax = jnp.max(jnp.abs(x))
    g = jnp.where(amax > 0, amax / (_E2M1_MAX * _E4M3_MAX), 1.0)
    y = x / g
    grouped = y.reshape(*y.shape[:-1], y.shape[-1] // block, block)
    sf = _e4m3_ceil(jnp.max(jnp.abs(grouped), axis=-1) / _E2M1_MAX)
    rounded = _e2m1_nearest(grouped / sf[..., None]) * sf[..., None]
    return (rounded.reshape(y.shape) * g).astype(x.dtype)


def _quantize_axis(x: jax.Array, axis: int, block: int = _BLOCK) -> jax.Array:
    """Requantize a saved factor over a consumer GEMM axis.

    NVFP4 block scales belong to the contraction axis of each product.  A
    factor saved for one forward contraction therefore must be decoded and
    regrouped when a backward product contracts along another axis.  This
    reference uses nearest rounding; a native implementation supplies the
    independent stochastic streams for each operand.
    """
    axis = axis % x.ndim
    return jnp.moveaxis(
        _quantize_last_axis(jnp.moveaxis(x, axis, -1), block), -1, axis
    )


def _quantized_einsum(
    equation: str,
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    lhs_axis: int,
    rhs_axis: int,
) -> jax.Array:
    """Reference FP4-operand product with consumer-axis recasts.

    The CUDA implementation uses independent stochastic streams for the two
    operands.  This portable JAX backend deliberately uses deterministic
    nearest rounding (JAX's stateless random keys are not threaded through
    the custom VJP); callers should not mistake it for the final estimator.
    """
    return jnp.einsum(
        equation,
        _quantize_axis(lhs, lhs_axis),
        _quantize_axis(rhs, rhs_axis),
        precision=jax.lax.Precision.HIGHEST,
    )


def _quantize_v(x: jax.Array, block: int = _BLOCK) -> jax.Array:
    """Decode NVFP4 V factors with scale blocks along the sequence axis."""
    # BSHD -> BHD S, quantize along S, then return BSHD.
    b, s, h, d = x.shape
    if s % block:
        raise ValueError(f"sequence length {s} must be divisible by {block}")
    y = jnp.transpose(x.astype(jnp.float32), (0, 2, 3, 1))
    y = _quantize_last_axis(y, block)
    return jnp.transpose(y, (0, 3, 1, 2))


def _quantize_numerator(u: jax.Array, block: int = _BLOCK) -> jax.Array:
    """Nearest NVFP4 decode of the normalized softmax numerator ``U``.

    CroissAttn uses one E4M3 scale per 16-key block for this factor.  ``U``
    is already normalized by the row maximum, so it does not use the tensor
    multiplier used by Q/K/V.
    """
    if u.shape[-1] % block:
        raise ValueError(f"key length {u.shape[-1]} must be divisible by {block}")
    uf = u.astype(jnp.float32)
    grouped = uf.reshape(*uf.shape[:-1], uf.shape[-1] // block, block)
    sf = _e4m3_ceil(jnp.max(grouped, axis=-1) / _E2M1_MAX)
    rounded = _e2m1_nearest(grouped / sf[..., None]) * sf[..., None]
    return rounded.reshape(uf.shape)


def _attention_forward(
    q,
    k,
    v,
    scale: float,
    causal: bool,
    *,
    return_dense: bool = True,
):
    """Compute quantized attention.

    ``return_dense`` is kept for the small numerical tests, which inspect the
    probability matrix directly.  The custom-VJP path sets it to ``False``:
    only the quantized factors and row statistics cross the forward/backward
    boundary.  Backward reconstructs probability and ``O_aux`` tiles from
    those saved payloads, matching the bounded-workspace CroissAttn contract.
    """
    qf = _quantize_last_axis(q)
    kf = _quantize_last_axis(k)
    vf = _quantize_v(v)
    # BSHD -> BHSD; GQA heads are expanded to query heads.
    qh = jnp.transpose(qf, (0, 2, 1, 3))
    kh = jnp.transpose(kf, (0, 2, 1, 3))
    vh = jnp.transpose(vf, (0, 2, 1, 3))
    if qh.shape[1] % kh.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    groups = qh.shape[1] // kh.shape[1]
    if groups != 1:
        kh = jnp.repeat(kh, groups, axis=1)
        vh = jnp.repeat(vh, groups, axis=1)
    logits = jnp.einsum("bhid,bhjd->bhij", qh, kh, precision=jax.lax.Precision.HIGHEST) * scale
    # Align the causal diagonal for rectangular query/key lengths.  The
    # decoder mask permits the final ``sk - sq`` keys when a shorter query
    # suffix attends into a longer KV prefix; ``jnp.tril`` would incorrectly
    # discard those keys by assuming ``sq == sk``.
    sq, sk = logits.shape[-2:]
    rows = jnp.arange(sq, dtype=jnp.int32)[:, None]
    cols = jnp.arange(sk, dtype=jnp.int32)[None, :]
    mask = cols <= rows + (sk - sq)
    causal_logits = jnp.where(mask, logits, -jnp.inf)
    logits = jnp.where(jnp.asarray(causal), causal_logits, logits)
    # CroissAttn's two-pass softmax keeps the exact numerator for the
    # derivative correction but rounds a separate numerator for public O.
    logits2 = logits * jnp.asarray(_LOG2E, dtype=jnp.float32)
    m2 = jnp.max(logits2, axis=-1, keepdims=True)
    u = jnp.exp2(logits2 - m2)
    uf = _quantize_numerator(u)
    l = jnp.sum(u, axis=-1, keepdims=True)
    pf = uf / l
    out_bhsd = jnp.einsum("bhij,bhjd->bhid", pf, vh, precision=jax.lax.Precision.HIGHEST)
    out = jnp.transpose(out_bhsd, (0, 2, 1, 3)).astype(q.dtype)
    if return_dense:
        # Keep O_aux in FP32 for the direct numerical tests.  The custom-VJP
        # path replays it in tiles below and never retains an SxS residual.
        p = u / l
        o_aux = jnp.einsum("bhij,bhjd->bhid", p, vh, precision=jax.lax.Precision.HIGHEST)
        return out, (qh, kh, vh, p, pf, o_aux, groups)
    # Row statistics are enough to replay P exactly from the saved Q/K
    # payloads.  Save the maximum in log2 space because U is formed with
    # exp2; keeping this convention avoids a second exp/log conversion in
    # every backward tile.  Pad only the replay residual to a static tile
    # multiple.  The public result remains the original SxD shape.
    m2 = m2.astype(jnp.float32)
    sq, sk = qh.shape[-2], kh.shape[-2]
    tile = min(_REPLAY_TILE, sq, sk)
    sqp = ((sq + tile - 1) // tile) * tile
    skp = ((sk + tile - 1) // tile) * tile
    qhp = jnp.pad(qh, ((0, 0), (0, 0), (0, sqp - sq), (0, 0)))
    khp = jnp.pad(kh, ((0, 0), (0, 0), (0, skp - sk), (0, 0)))
    vhp = jnp.pad(vh, ((0, 0), (0, 0), (0, skp - sk), (0, 0)))
    ufp = jnp.pad(uf, ((0, 0), (0, 0), (0, sqp - sq), (0, skp - sk)))
    lp = jnp.pad(l, ((0, 0), (0, 0), (0, sqp - sq), (0, 0)), constant_values=1.0)
    mp = jnp.pad(m2, ((0, 0), (0, 0), (0, sqp - sq), (0, 0)), constant_values=0.0)
    return out, (qhp, khp, vhp, ufp, lp, mp, groups, sq, sk, tile)


def _pad_bhsd(x: jax.Array, sequence: int) -> jax.Array:
    """Pad a BHSD tensor along sequence to a statically tiled length."""
    extra = sequence - x.shape[-2]
    if extra <= 0:
        return x
    return jnp.pad(x, ((0, 0), (0, 0), (0, extra), (0, 0)))


def _pad_bhss(x: jax.Array, sequence: int) -> jax.Array:
    """Pad a BHSS tensor in both sequence axes to a static tile multiple."""
    extra = sequence - x.shape[-2]
    if extra <= 0:
        return x
    return jnp.pad(x, ((0, 0), (0, 0), (0, extra), (0, extra)))


def _replay_probability_tile(
    qh: jax.Array,
    kh: jax.Array,
    row_max2: jax.Array,
    denominator: jax.Array,
    scale: jax.Array,
    causal: jax.Array,
    q0: jax.Array,
    k0: jax.Array,
    tile: int,
    sq: int,
    sk: int,
) -> jax.Array:
    """Replay one causal P tile from saved quantized Q/K payloads.

    All slice sizes are static, so this helper lowers to bounded dynamic
    slices under ``jit``.  ``q0`` and ``k0`` remain loop indices and never
    trigger host-side concretization.
    """
    b, h, _, d = qh.shape
    qt = jax.lax.dynamic_slice(qh, (0, 0, q0, 0), (b, h, tile, d))
    kt = jax.lax.dynamic_slice(kh, (0, 0, k0, 0), (b, h, tile, d))
    scores2 = jnp.einsum(
        "bhid,bhjd->bhij", qt, kt, precision=jax.lax.Precision.HIGHEST
    ) * scale * jnp.asarray(_LOG2E, dtype=jnp.float32)
    rows = q0 + jnp.arange(tile, dtype=jnp.int32)
    cols = k0 + jnp.arange(tile, dtype=jnp.int32)
    valid = (rows[:, None] < sq) & (cols[None, :] < sk)
    valid = valid & jnp.where(causal, cols[None, :] <= rows[:, None] + (sk - sq), True)
    scores2 = jnp.where(valid[None, None, :, :], scores2, -jnp.inf)
    mt = jax.lax.dynamic_slice(row_max2, (0, 0, q0, 0), (b, h, tile, 1))
    lt = jax.lax.dynamic_slice(denominator, (0, 0, q0, 0), (b, h, tile, 1))
    return jnp.exp2(scores2 - mt) / lt


@jax.custom_vjp
def croissattn_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> jax.Array:
    """Reference CroissAttn forward with corrected backward.

    Inputs use ``[batch, sequence, heads, head_dim]``.  Q/K head dimensions
    must be 128 and sequence/head-dimension blocks must be divisible by 16.
    The implementation is intended for correctness tests and small canaries;
    it is not a replacement for the fused SM100 kernels.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    return _attention_forward(q, k, v, jnp.asarray(softmax_scale, dtype=jnp.float32), causal)[0]


def _fwd(q, k, v, softmax_scale=None, causal=True):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    out, saved = _attention_forward(
        q,
        k,
        v,
        jnp.asarray(softmax_scale, dtype=jnp.float32),
        causal,
        return_dense=False,
    )
    return out, (saved, jnp.asarray(softmax_scale, dtype=jnp.float32), causal)


def _bwd(res, dout):
    (qh, kh, vh, uf, l, row_max2, groups, sq, sk, tile), scale, causal = res
    # dO is BSHD; move it to BHSD and accumulate the softmax derivative using
    # the unrounded O_aux.  O_aux and all score tiles are replayed from the
    # saved quantized payloads, so the residual has no dense SxS matrix.
    doh = jnp.transpose(dout, (0, 2, 1, 3)).astype(jnp.float32)
    b, h, _, d = qh.shape
    doh_p = jnp.pad(doh, ((0, 0), (0, 0), (0, qh.shape[-2] - sq), (0, 0)))
    # Recast each saved factor once over its *full* consumer contraction axis
    # before slicing tiles.  Requantizing every tile independently would make
    # block scales depend on the tile boundary and is the subtle small/large
    # block bug this reference is intended to expose.
    q_rec = _quantize_axis(qh[..., :sq, :], -2)
    k_rec = _quantize_axis(kh[..., :sk, :], -2)
    v_rec = _quantize_axis(vh[..., :sk, :], -1)
    do_d_rec = _quantize_axis(doh, -1)
    dn_rec = _quantize_axis(doh / l[..., :sq, :], -2)
    u_q_rec = _quantize_axis(uf[..., :sq, :sk], -2)
    q_rec = jnp.pad(q_rec, ((0, 0), (0, 0), (0, qh.shape[-2] - sq), (0, 0)))
    k_rec = jnp.pad(k_rec, ((0, 0), (0, 0), (0, kh.shape[-2] - sk), (0, 0)))
    v_rec = jnp.pad(v_rec, ((0, 0), (0, 0), (0, vh.shape[-2] - sk), (0, 0)))
    do_d_rec = jnp.pad(do_d_rec, ((0, 0), (0, 0), (0, qh.shape[-2] - sq), (0, 0)))
    dn_rec = jnp.pad(dn_rec, ((0, 0), (0, 0), (0, qh.shape[-2] - sq), (0, 0)))
    u_q_rec = jnp.pad(u_q_rec, ((0, 0), (0, 0), (0, qh.shape[-2] - sq), (0, kh.shape[-2] - sk)))
    dstar = jnp.zeros((b, h, qh.shape[-2], 1), dtype=jnp.float32)

    def replay_oaux(qi, acc):
        q0 = qi * tile
        p_i = jnp.zeros((b, h, tile, tile), dtype=jnp.float32)
        o_i = jnp.zeros((b, h, tile, d), dtype=jnp.float32)

        def replay_key(ki, state):
            pacc, oacc = state
            k0 = ki * tile
            p_ij = _replay_probability_tile(
                qh,
                kh,
                row_max2,
                l,
                scale,
                causal,
                q0,
                k0,
                tile,
                sq,
                sk,
            )
            v_j = jax.lax.dynamic_slice(vh, (0, 0, k0, 0), (b, h, tile, d))
            return pacc, oacc + jnp.einsum(
                "bhij,bhjd->bhid", p_ij, v_j, precision=jax.lax.Precision.HIGHEST
            )

        _, o_i = jax.lax.fori_loop(0, kh.shape[-2] // tile, replay_key, (p_i, o_i))
        do_i = jax.lax.dynamic_slice(doh_p, (0, 0, q0, 0), (b, h, tile, d))
        dsi = jnp.sum(do_i * o_i, axis=-1, keepdims=True)
        return jax.lax.dynamic_update_slice(acc, dsi, (0, 0, q0, 0))

    dstar = jax.lax.fori_loop(0, qh.shape[-2] // tile, replay_oaux, dstar)

    dqh = jnp.zeros_like(qh, dtype=jnp.float32)
    dkh = jnp.zeros_like(kh, dtype=jnp.float32)
    dvh = jnp.zeros_like(vh, dtype=jnp.float32)
    dn = doh_p / l

    def replay_query(qi, state):
        dq_acc, dk_acc, dv_acc = state
        q0 = qi * tile
        q_i = jax.lax.dynamic_slice(qh, (0, 0, q0, 0), (b, h, tile, d))
        do_i = jax.lax.dynamic_slice(doh_p, (0, 0, q0, 0), (b, h, tile, d))
        ds_i = jax.lax.dynamic_slice(dstar, (0, 0, q0, 0), (b, h, tile, 1))
        dn_i = jax.lax.dynamic_slice(dn_rec, (0, 0, q0, 0), (b, h, tile, d))

        def replay_key(ki, state2):
            dq_a, dk_a, dv_a = state2
            k0 = ki * tile
            p_ij = _replay_probability_tile(
                qh,
                kh,
                row_max2,
                l,
                scale,
                causal,
                q0,
                k0,
                tile,
                sq,
                sk,
            )
            k_j = jax.lax.dynamic_slice(k_rec, (0, 0, k0, 0), (b, h, tile, d))
            v_j = jax.lax.dynamic_slice(v_rec, (0, 0, k0, 0), (b, h, tile, d))
            u_ij = jax.lax.dynamic_slice(u_q_rec, (0, 0, q0, k0), (b, h, tile, tile))
            do_d_i = jax.lax.dynamic_slice(do_d_rec, (0, 0, q0, 0), (b, h, tile, d))
            # dO V^T uses the D-axis recasts prepared above.
            dp_ij = jnp.einsum(
                "bhid,bhjd->bhij", do_d_i, v_j, precision=jax.lax.Precision.HIGHEST
            )
            ds_ij = p_ij * (dp_ij - ds_i)
            # dS K and dS^T Q use consumer-axis recasts for every saved
            # factor.  This is the fix that prevents silently reusing a
            # factor quantized on the wrong (forward) axis.
            ds_q = _quantize_axis(ds_ij, -1)
            ds_k = _quantize_axis(ds_ij, -2)
            dq_delta = jnp.einsum(
                "bhij,bhjd->bhid", ds_q, k_j, precision=jax.lax.Precision.HIGHEST
            ) * scale
            q_j = jax.lax.dynamic_slice(q_rec, (0, 0, q0, 0), (b, h, tile, d))
            dk_delta = jnp.einsum(
                "bhij,bhid->bhjd", ds_k, q_j, precision=jax.lax.Precision.HIGHEST
            ) * scale
            # U^T (L^-1 dO), with dO normalized before quantization.
            dv_delta = jnp.einsum(
                "bhij,bhid->bhjd", u_ij, dn_i, precision=jax.lax.Precision.HIGHEST
            )
            # q0/k0 are loop-carried tracers.  NumPy-style slices cannot use
            # them under jit; dynamic_update_slice keeps the tile bounded and
            # lets XLA lower the add without materialising an SxS tensor.
            dq_a = dq_a + jax.lax.dynamic_update_slice(
                jnp.zeros_like(dq_a), dq_delta, (0, 0, q0, 0)
            )
            dk_a = dk_a + jax.lax.dynamic_update_slice(
                jnp.zeros_like(dk_a), dk_delta, (0, 0, k0, 0)
            )
            dv_a = dv_a + jax.lax.dynamic_update_slice(
                jnp.zeros_like(dv_a), dv_delta, (0, 0, k0, 0)
            )
            return dq_a, dk_a, dv_a

        return jax.lax.fori_loop(0, kh.shape[-2] // tile, replay_key, (dq_acc, dk_acc, dv_acc))

    dqh, dkh, dvh = jax.lax.fori_loop(
        0, qh.shape[-2] // tile, replay_query, (dqh, dkh, dvh)
    )
    d_q = jnp.transpose(dqh[..., :sq, :], (0, 2, 1, 3)).astype(dout.dtype)
    dkh = dkh[..., :sk, :]
    dvh = dvh[..., :sk, :]
    if groups != 1:
        # GQA expansion is a repeat in forward; collapse the corresponding
        # query-head groups before returning the KV gradients.
        hq = dkh.shape[1]
        hkv = hq // groups
        dkh = dkh.reshape(b, hkv, groups, sk, d).sum(axis=2)
        dvh = dvh.reshape(b, hkv, groups, sk, d).sum(axis=2)
    d_k = jnp.transpose(dkh, (0, 2, 1, 3)).astype(dout.dtype)
    d_v = jnp.transpose(dvh, (0, 2, 1, 3)).astype(dout.dtype)
    return d_q, d_k, d_v, None, None


croissattn_reference.defvjp(_fwd, _bwd)


__all__ = ["croissattn_reference"]
